import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from src.data import CarConditionDataset, DataConfig
from src.train import validate_metrics


def save_confusion(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(3,3), dpi=120)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1], labels)
    ax.set_yticks([0,1], labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-csv', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--device', default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    ap.add_argument('--out-dir', default='eval')
    ap.add_argument('--backbone', default='auto', choices=['auto', 'resnet18', 'resnet34', 'resnet50'], help='Бэкбон для инициализации модели. По умолчанию auto — подобрать по чекпойнту.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Подгружаем предсказания через train.validate_metrics (повторная валидация)
    from src.model import MultiHeadResNet
    import torch
    from torch.utils.data import DataLoader

    ds = CarConditionDataset(args.val_csv, DataConfig(image_size=args.image_size, augment=False))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    state = torch.load(args.ckpt, map_location=args.device)

    # Подбираем правильный бэкбон под чекпойнт
    def try_load(backbone_name: str):
        m = MultiHeadResNet(pretrained=False, backbone=backbone_name)
        m.load_state_dict(state['model'])
        return m

    model = None
    if args.backbone != 'auto':
        model = try_load(args.backbone)
    else:
        last_err = None
        for bb in ['resnet18', 'resnet34', 'resnet50']:
            try:
                model = try_load(bb)
                print(f"Loaded checkpoint with backbone={bb}")
                break
            except Exception as e:
                last_err = e
                continue
        if model is None:
            raise last_err
    model.to(args.device)

    metrics = validate_metrics(model, dl, device=args.device)
    print(metrics)

    # Для матриц ошибок соберём y и p
    ys, ps, ps_proba = [], [], []
    # Пороги из чекпойнта (если сохранены в train.py)
    thr_clean = state.get('thresholds', {}).get('clean', 0.5) if isinstance(state, dict) else 0.5
    thr_dmg = state.get('thresholds', {}).get('damaged', 0.5) if isinstance(state, dict) else 0.5
    model.eval()
    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(args.device)
            lc, ld = model(x)
            pc_scores = lc.sigmoid().squeeze(1).cpu().numpy()
            pd_scores = ld.sigmoid().squeeze(1).cpu().numpy()
            pc = (pc_scores >= thr_clean).astype(int)
            pd = (pd_scores >= thr_dmg).astype(int)
            ys.append(y.numpy()); ps.append(np.stack([pc, pd], axis=1)); ps_proba.append(np.stack([pc_scores, pd_scores], axis=1))
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    s = np.concatenate(ps_proba, axis=0)

    save_confusion(y[:,0], p[:,0], ["dirty","clean"], 'Clean vs Dirty', os.path.join(args.out_dir, 'cm_clean.png'))
    save_confusion(y[:,1], p[:,1], ["intact","damaged"], 'Damaged vs Intact', os.path.join(args.out_dir, 'cm_damaged.png'))
    print('Confusion matrices saved to', args.out_dir)

    # ROC & PR curves
    def plot_roc_pr(y_true, scores, title_prefix, out_prefix):
        auc = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)

        # ROC
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=120)
        ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
        ax.plot([0,1],[0,1],'k--', alpha=0.3)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(f'{title_prefix} ROC')
        ax.legend(loc='lower right')
        fig.tight_layout(); plt.savefig(os.path.join(args.out_dir, f'{out_prefix}_roc.png')); plt.close(fig)

        # PR
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=120)
        ax.plot(rec, prec, label=f'AP={ap:.3f}')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title(f'{title_prefix} PR')
        ax.legend(loc='lower left')
        fig.tight_layout(); plt.savefig(os.path.join(args.out_dir, f'{out_prefix}_pr.png')); plt.close(fig)

    plot_roc_pr(y[:,0], s[:,0], 'Clean', 'clean')
    plot_roc_pr(y[:,1], s[:,1], 'Damaged', 'damaged')
    print('ROC and PR curves saved to', args.out_dir)

if __name__ == '__main__':
    main()
