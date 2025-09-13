import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='CSV с колонками: filepath,clean,damaged')
    ap.add_argument('--val-size', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='data/splits')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.labels)
    assert {'filepath', 'clean', 'damaged'}.issubset(df.columns)

    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed, stratify=df[['clean','damaged']])
    train_df.to_csv(os.path.join(args.out_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(args.out_dir, 'val.csv'), index=False)
    print('Saved splits to', args.out_dir)


if __name__ == '__main__':
    main()
