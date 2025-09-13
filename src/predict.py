import os
import glob
from typing import List, Optional

import tyro
from dataclasses import dataclass
import torch
from PIL import Image
from torchvision import transforms

from .model import MultiHeadResNet


def load_image(path: str, size: int = 224):
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return t(img).unsqueeze(0)


def soft_labels(pc: float, pd: float, thr_c: float = 0.5, thr_d: float = 0.5):
    return {
        'clean_prob': pc,
        'dirty_prob': 1 - pc,
        'intact_prob': 1 - pd,
    'damaged_prob': pd,
    'pred_clean': 'clean' if pc >= thr_c else 'dirty',
    'pred_damage': 'damaged' if pd >= thr_d else 'intact',
    }


@dataclass
class PredictConfig:
    ckpt: str
    image: Optional[str] = None
    folder: Optional[str] = None
    image_size: int = 224
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone: str = 'resnet18'


def predict_one(ckpt: str, image_path: str, image_size: int, device: str, backbone: str):
    state = torch.load(ckpt, map_location=device)
    model = MultiHeadResNet(pretrained=False, backbone=backbone)
    model.load_state_dict(state['model'])
    model.to(device).eval()
    x = load_image(image_path, size=image_size).to(device)
    with torch.no_grad():
        pc, pd = model.predict_proba(x)
        pc = pc.item()
        pd = pd.item()
    thr_c = state.get('thresholds', {}).get('clean', 0.5)
    thr_d = state.get('thresholds', {}).get('damaged', 0.5)
    return soft_labels(pc, pd, thr_c, thr_d)


def main(cfg: PredictConfig):
    paths: List[str] = []
    if cfg.image:
        paths = [cfg.image]
    elif cfg.folder:
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        for p in patterns:
            paths.extend(glob.glob(os.path.join(cfg.folder, p)))
    else:
        raise SystemExit("Укажите --image или --folder")

    for p in paths:
        res = predict_one(cfg.ckpt, p, cfg.image_size, cfg.device, cfg.backbone)
        print(p, res)


if __name__ == '__main__':
    cfg = tyro.cli(PredictConfig)
    main(cfg)
