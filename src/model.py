from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


def _make_backbone(name: str, pretrained: bool):
    name = name.lower()
    if name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        m = resnet18(weights=weights)
    elif name == 'resnet34':
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        m = resnet34(weights=weights)
    elif name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        m = resnet50(weights=weights)
    else:
        raise ValueError(f'Unknown backbone: {name}')
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim


class MultiHeadResNet(nn.Module):
    """
    Backbone: ResNet18 (замороженные/частично замороженные слои опционально)
    Heads: две бинарные головы для (clean, damaged)
    Выход: logits_clean [B,1], logits_damaged [B,1]
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False, backbone: str = 'resnet18'):
        super().__init__()
        self.backbone, feat_dim = _make_backbone(backbone, pretrained)

        self.head_clean = nn.Linear(feat_dim, 1)
        self.head_damaged = nn.Linear(feat_dim, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)  # [B, feat]
        clean = self.head_clean(feats)
        damaged = self.head_damaged(feats)
        return clean, damaged

    def predict_proba(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            lc, ld = self.forward(x)
            pc = torch.sigmoid(lc)
            pd = torch.sigmoid(ld)
        return pc, pd
