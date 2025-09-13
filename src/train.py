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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def bce_loss(logits, targets):
    return nn.BCEWithLogitsLoss()(logits, targets)


def run_epoch(model, loader: DataLoader, optimizer=None, device='cpu'):
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter = 0.0
    n = 0

    for x, y, _ in tqdm(loader, desc='train' if is_train else 'val'):
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        lc, ld = model(x)
        loss = bce_loss(lc.squeeze(1), y[:, 0]) + bce_loss(ld.squeeze(1), y[:, 1])
        if is_train:
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
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    dcfg = DataConfig(image_size=cfg.image_size, augment=True)
    train_loader = make_loader(cfg.train_csv, dcfg, batch_size=cfg.batch_size, shuffle=True)
    val_loader = make_loader(cfg.val_csv, DataConfig(image_size=cfg.image_size, augment=False), batch_size=cfg.batch_size, shuffle=False)

    model = MultiHeadResNet(pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone).to(cfg.device)

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_f1 = -1.0
    best_path = cfg.out

    for epoch in range(1, cfg.epochs + 1):
        print(f"Epoch {epoch}/{cfg.epochs}")
        train_loss = run_epoch(model, train_loader, optimizer=opt, device=cfg.device)
        val_loss = run_epoch(model, val_loader, optimizer=None, device=cfg.device)
        metrics = validate_metrics(model, val_loader, device=cfg.device)
        sched.step()
        print({
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'metrics': format_metrics(metrics),
        })
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            torch.save({'model': model.state_dict(), 'metrics': metrics, 'epoch': epoch}, best_path)
            print(f"Saved best to {best_path} with macro_f1={best_f1:.4f}")


if __name__ == '__main__':
    cfg = tyro.cli(TrainConfig)
    main(cfg)
