import os
import math
from dataclasses import dataclass
from typing import Optional

import tyro
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import make_loader, DataConfig
from .model import MultiHeadResNet
from .utils import format_metrics


@dataclass
class TrainConfig:
    train_csv: str
    val_csv: str
    out: str = 'checkpoints/resnet18_multitask.pt'
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    pretrained: bool = True
    freeze_backbone: bool = False
    freeze_epochs: int = 0
    backbone: str = 'resnet18'
    workers: int = 2
    amp: bool = True
    early_stop_patience: int = 5
    seed: int = 42
    log_csv: Optional[str] = 'checkpoints/train_log.csv'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_bce_loss(pos_weight: Optional[torch.Tensor] = None):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def run_epoch(model, loader: DataLoader, optimizer=None, device='cpu', scaler: Optional[torch.cuda.amp.GradScaler] = None, losses=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter = 0.0
    n = 0

    for x, y, _ in tqdm(loader, desc='train' if is_train else 'val'):
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            lc, ld = model(x)
            loss_c = losses['clean'](lc.squeeze(1), y[:, 0])
            loss_d = losses['damaged'](ld.squeeze(1), y[:, 1])
            loss = loss_c + loss_d
        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        loss_meter += loss.item() * x.size(0)
        n += x.size(0)
    return loss_meter / max(n, 1)


def validate_metrics(model, loader: DataLoader, device='cpu'):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc='metrics'):
            x = x.to(device)
            lc, ld = model(x)
            pc = torch.sigmoid(lc).squeeze(1).cpu()
            pd = torch.sigmoid(ld).squeeze(1).cpu()
            ys.append(y)
            ps.append(torch.stack([pc, pd], dim=1))
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    def binarize(a, thr=0.5):
        return (a >= thr).astype('int')

    metrics = {}
    for i, name in enumerate(['clean', 'damaged']):
        yi = y[:, i]
        pi = p[:, i]
        pi_b = binarize(pi)
        acc = accuracy_score(yi, pi_b)
        prec, rec, f1, _ = precision_recall_fscore_support(yi, pi_b, average='binary', zero_division=0)
        metrics[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
    metrics['macro_f1'] = (metrics['clean']['f1'] + metrics['damaged']['f1']) / 2
    return metrics


def main(cfg: TrainConfig):
    # seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    import random, numpy as np
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    dcfg = DataConfig(image_size=cfg.image_size, augment=True)
    train_loader = make_loader(cfg.train_csv, dcfg, batch_size=cfg.batch_size, shuffle=True, workers=cfg.workers)
    val_loader = make_loader(cfg.val_csv, DataConfig(image_size=cfg.image_size, augment=False), batch_size=cfg.batch_size, shuffle=False, workers=cfg.workers)

    model = MultiHeadResNet(pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone or cfg.freeze_epochs>0, backbone=cfg.backbone).to(cfg.device)

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_f1 = -1.0
    best_path = cfg.out
    no_improve = 0
    # class imbalance handling: compute pos_weight from train CSV
    try:
        import pandas as pd
        df = pd.read_csv(cfg.train_csv)
        def pw(col):
            pos = max(df[col].sum(), 1)
            neg = max(len(df) - pos, 1)
            return torch.tensor([neg/pos], device=cfg.device)
        losses = {
            'clean': make_bce_loss(pos_weight=pw('clean')),
            'damaged': make_bce_loss(pos_weight=pw('damaged')),
        }
    except Exception:
        losses = {
            'clean': make_bce_loss(),
            'damaged': make_bce_loss(),
        }

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and 'cuda' in cfg.device and torch.cuda.is_available()))

    # logging csv
    log_f = None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv), exist_ok=True)
        log_f = open(cfg.log_csv, 'w', encoding='utf-8')
        print('epoch,train_loss,val_loss,clean_f1,damaged_f1,macro_f1', file=log_f, flush=True)

    for epoch in range(1, cfg.epochs + 1):
        # временная заморозка бэкбона на первые freeze_epochs
        if cfg.freeze_epochs>0 and epoch==1:
            for p in model.backbone.parameters():
                p.requires_grad = False
        if cfg.freeze_epochs>0 and epoch==cfg.freeze_epochs+1:
            for p in model.backbone.parameters():
                p.requires_grad = True
        print(f"Epoch {epoch}/{cfg.epochs}")
        train_loss = run_epoch(model, train_loader, optimizer=opt, device=cfg.device, scaler=scaler, losses=losses)
        val_loss = run_epoch(model, val_loader, optimizer=None, device=cfg.device, scaler=None, losses=losses)
        metrics = validate_metrics(model, val_loader, device=cfg.device)
        sched.step()
        print({
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'metrics': format_metrics(metrics),
        })
        if log_f:
            print(f"{epoch},{train_loss:.6f},{val_loss:.6f},{metrics['clean']['f1']:.6f},{metrics['damaged']['f1']:.6f},{metrics['macro_f1']:.6f}", file=log_f, flush=True)
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            # подберём пороги по валидации (максимум f1) и сохраним в чекпойнт
            def tune_thr(name_idx):
                import numpy as np
                y_scores, y_true = [], []
                model.eval()
                with torch.no_grad():
                    for x, y, _ in val_loader:
                        x = x.to(cfg.device)
                        lc, ld = model(x)
                        s = torch.sigmoid(lc if name_idx==0 else ld).squeeze(1).cpu().numpy()
                        y_scores.append(s)
                        y_true.append(y[:, name_idx].numpy())
                ys = np.concatenate(y_scores); yt = np.concatenate(y_true)
                from sklearn.metrics import f1_score
                best_f, best_t = -1, 0.5
                for t in np.linspace(0.1,0.9,33):
                    pred = (ys>=t).astype(int)
                    f = f1_score(yt, pred, zero_division=0)
                    if f>best_f:
                        best_f, best_t = f, t
                return float(best_t)

            thr_clean = tune_thr(0)
            thr_dmg = tune_thr(1)
            torch.save({'model': model.state_dict(), 'metrics': metrics, 'epoch': epoch, 'thresholds': {'clean': thr_clean, 'damaged': thr_dmg}}, best_path)
            print(f"Saved best to {best_path} with macro_f1={best_f1:.4f}")
            no_improve = 0
        else:
            no_improve += 1
        # early stopping
        if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
            print('Early stopping triggered.')
            break

    if log_f:
        log_f.close()


if __name__ == '__main__':
    cfg = tyro.cli(TrainConfig)
    main(cfg)
