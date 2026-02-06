"""
二分类网络：使用预训练的 ResNet-18 进行迁移学习。
"""
import torch
import torch.nn as nn
from torchvision import models

class PatchCNN(nn.Module):
    """基于 ResNet-18 的二分类模型"""

    def __init__(self, in_channels=3, num_classes=2, pretrained=True):
        super().__init__()
        # 加载 ResNet18
        # weights='DEFAULT' 对应最新的 ImageNet 权重 (ResNet18_Weights.IMAGENET1K_V1)
        # 如果 pytorch 版本较老不支持 weights 参数，可退回到 pretrained=True
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            # 兼容旧版本 torchvision
            self.backbone = models.resnet18(pretrained=pretrained)
        
        # 修改第一层（如果输入通道不是3）
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # 修改最后一层全连接层
        # ResNet18 的 fc 输入特征数是 512
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes=2, **kwargs):
    # kwargs 可以传递 pretrained=True/False
    return PatchCNN(num_classes=num_classes, **kwargs)
