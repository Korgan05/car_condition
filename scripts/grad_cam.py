import os
from dataclasses import dataclass

import numpy as np
import torch
import tyro
from PIL import Image
import cv2

from src.model import MultiHeadResNet
from src.app import build_transform


@dataclass
class CamConfig:
    ckpt: str = 'checkpoints/resnet18_multitask.pt'
    image: str = 'data/raw/dummy_000.png'
    out: str = 'eval/gradcam.png'
    image_size: int = 224
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    head: str = 'damaged'  # 'clean' | 'damaged'


def get_last_conv(model: MultiHeadResNet):
    # Для ResNet18 — layer4[-1].conv2
    return model.backbone.layer4[-1].conv2


def compute_cam(model: MultiHeadResNet, img_t: torch.Tensor, head: str):
    feats = None
    grads = None

    def fwd_hook(module, inp, out):
        nonlocal feats
        feats = out.detach()

    def bwd_hook(module, gin, gout):
        nonlocal grads
        grads = gout[0].detach()

    conv = get_last_conv(model)
    h1 = conv.register_forward_hook(fwd_hook)
    h2 = conv.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    lc, ld = model(img_t)
    logit = lc if head == 'clean' else ld
    logit.backward(torch.ones_like(logit))

    h1.remove(); h2.remove()
    # GAP по градиентам как веса
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
    cam = (weights * feats).sum(dim=1, keepdim=False)  # [B,H,W]
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam[0].cpu().numpy()


def overlay_cam(img: np.ndarray, cam: np.ndarray):
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (0.4 * heat + 0.6 * img).astype(np.uint8)
    return out


def main(cfg: CamConfig):
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)

    model = MultiHeadResNet(pretrained=False)
    state = torch.load(cfg.ckpt, map_location=cfg.device)
    model.load_state_dict(state['model'])
    model.to(cfg.device).eval()

    t = build_transform(cfg.image_size)
    img = Image.open(cfg.image).convert('RGB')
    img_np = np.array(img)
    x = t(img).unsqueeze(0).to(cfg.device)

    with torch.enable_grad():
        cam = compute_cam(model, x, cfg.head)

    vis = overlay_cam(img_np, cam)
    Image.fromarray(vis).save(cfg.out)
    print('Saved Grad-CAM to', cfg.out)


if __name__ == '__main__':
    cfg = tyro.cli(CamConfig)
    main(cfg)
