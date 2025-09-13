import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class DataConfig:
    image_size: int = 224
    augment: bool = True


def build_transforms(cfg: DataConfig):
    t = []
    if cfg.augment:
        t += [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]
    else:
        t += [transforms.Resize((cfg.image_size, cfg.image_size))]
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(t)


class CarConditionDataset(Dataset):
    def __init__(self, csv_path: str, cfg: Optional[DataConfig] = None):
        self.df = pd.read_csv(csv_path)
        assert {'filepath', 'clean', 'damaged'}.issubset(self.df.columns), (
            'CSV должен содержать столбцы: filepath, clean, damaged'
        )
        self.cfg = cfg or DataConfig()
        self.transform = build_transforms(self.cfg)
        # сохраняем путь к csv, чтобы уметь резолвить относительные пути к изображениям
        self._csv_path = os.path.abspath(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row['filepath']
        # интерпретируем относительные пути с устойчивым резолвингом
        img_path = self._resolve_path(path)

        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        y_clean = torch.tensor(int(row['clean']), dtype=torch.float32)
        y_damaged = torch.tensor(int(row['damaged']), dtype=torch.float32)
        y = torch.stack([y_clean, y_damaged])  # shape [2]
        return x, y, img_path

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        base = os.path.dirname(self._csv_path) if getattr(self, '_csv_path', None) else os.getcwd()
        cand = os.path.abspath(os.path.join(base, p))
        if os.path.exists(cand):
            return cand
        # поднимаемся до 3 уровней вверх (на случай data/splits/* -> корень проекта)
        cur = base
        for _ in range(3):
            cur = os.path.dirname(cur)
            cand = os.path.abspath(os.path.join(cur, p))
            if os.path.exists(cand):
                return cand
        # последний шанс — от CWD
        return os.path.abspath(os.path.join(os.getcwd(), p))


def make_loader(csv_path: str, cfg: Optional[DataConfig], batch_size: int, shuffle: bool, workers: int = 2, pin_memory: bool = True) -> DataLoader:
    ds = CarConditionDataset(csv_path, cfg)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=pin_memory)
