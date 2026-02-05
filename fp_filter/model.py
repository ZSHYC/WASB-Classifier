"""
二分类网络：轻量 CNN，输入为小 patch（如 32x32），输出 2 类（球/非球）。
"""
import torch
import torch.nn as nn


class PatchCNN(nn.Module):
    """轻量级 patch 二分类 CNN，适用于 32x32 输入。"""

    def __init__(self, in_channels=3, num_classes=2, base_channels=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=2, **kwargs):
    return PatchCNN(in_channels=3, num_classes=num_classes, **kwargs)
