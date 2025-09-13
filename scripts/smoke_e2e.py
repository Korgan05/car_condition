import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))


def run(cmd, cwd=None):
    print('> ', cmd)
    res = subprocess.run(cmd, cwd=cwd or ROOT, shell=True)
    if res.returncode != 0:
        raise SystemExit(f'Command failed: {cmd}')


def main():
    # 1) generate synthetic data
    run(f"{sys.executable} -m scripts.create_dummy_dataset")

    # 2) split
    run(f"{sys.executable} -m scripts.prepare_split --labels data/labels.csv --val-size 0.2")

    # 3) train 1 epoch
    ckpt = os.path.join('checkpoints', 'resnet18_multitask.pt')
    run(f"{sys.executable} -m src.train --train-csv data/splits/train.csv --val-csv data/splits/val.csv --epochs 1 --batch-size 8 --lr 3e-4 --out {ckpt}")

    # 4) predict one image
    # pick any generated image
    sample = os.path.join('data', 'raw', 'dummy_000.png')
    run(f"{sys.executable} -m src.predict --ckpt {ckpt} --image {sample}")

    print('Smoke E2E completed successfully.')


if __name__ == '__main__':
    main()
