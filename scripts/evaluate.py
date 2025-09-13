import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Подгружаем предсказания через train.validate_metrics (повторная валидация)
    from src.model import MultiHeadResNet
    import torch
    from torch.utils.data import DataLoader

    ds = CarConditionDataset(args.val_csv, DataConfig(image_size=args.image_size, augment=False))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    model = MultiHeadResNet(pretrained=False)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state['model'])
    model.to(args.device)

    metrics = validate_metrics(model, dl, device=args.device)
    print(metrics)

    # Для матриц ошибок соберём y и p
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(args.device)
            lc, ld = model(x)
            pc = (lc.sigmoid().squeeze(1).cpu().numpy() >= 0.5).astype(int)
            pd = (ld.sigmoid().squeeze(1).cpu().numpy() >= 0.5).astype(int)
            ys.append(y.numpy()); ps.append(np.stack([pc, pd], axis=1))
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)

    save_confusion(y[:,0], p[:,0], ["dirty","clean"], 'Clean vs Dirty', os.path.join(args.out_dir, 'cm_clean.png'))
    save_confusion(y[:,1], p[:,1], ["intact","damaged"], 'Damaged vs Intact', os.path.join(args.out_dir, 'cm_damaged.png'))
    print('Confusion matrices saved to', args.out_dir)

if __name__ == '__main__':
    main()
