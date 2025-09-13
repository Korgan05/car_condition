from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class MultiHeadResNet(nn.Module):
    """
    Backbone: ResNet18 (замороженные/частично замороженные слои опционально)
    Heads: две бинарные головы для (clean, damaged)
    Выход: logits_clean [B,1], logits_damaged [B,1]
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

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
