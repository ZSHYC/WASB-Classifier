"""
二分类数据集：读取 manifest.csv（含 patch_path 与 label 列），
label=1 为球(TP)，label=0 为非球(FP)。
"""
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


# 常用 ImageNet 归一化，与项目内一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PatchDataset(Dataset):
    """从 manifest CSV 加载 patch 图像与二分类标签。"""

    def __init__(self, manifest_path, transform=None, target_one_hot=False):
        """
        manifest_path: 第一步生成的 manifest.csv 路径（需已标注 label 列：1=球，0=非球）。
        transform: 可选，对 PIL/ndarray 的变换（若为 None 则仅 ToTensor + 归一化）。
        target_one_hot: 若 True，返回 one-hot [1,0]/[0,1]；否则返回 0/1 标量。
        """
        self.manifest_dir = osp.dirname(osp.abspath(manifest_path))
        self.df = pd.read_csv(manifest_path)
        # 只保留已标注行
        self.df = self.df[self.df["label"].notna() & (self.df["label"].astype(str).str.strip() != "")]
        self.df = self.df.astype({"label": int})
        if self.df["label"].min() < 0 or self.df["label"].max() > 1:
            raise ValueError("manifest 中 label 只能为 0 或 1")
        self.transform = transform
        self.target_one_hot = target_one_hot

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row["patch_path"]
        if not osp.isfile(path) and not osp.isabs(path):
            path = osp.join(self.manifest_dir, path)
        if not osp.isfile(path):
            raise FileNotFoundError(f"patch 文件不存在: {path}")
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"无法读取: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(row["label"])

        if self.transform is not None:
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(img)
            img = self.transform(img_pil)
        else:
            # 默认：ToTensor + 归一化
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            img = (img - mean) / std

        if self.target_one_hot:
            target = torch.zeros(2, dtype=torch.float32)
            target[label] = 1.0
        else:
            target = torch.tensor(label, dtype=torch.long)

        return img, target


def get_default_transform(is_train, patch_size=96):
    """默认数据增强：训练时随机水平翻转，测试时仅归一化。输入为 PIL 或 ndarray。"""
    import torchvision.transforms as T
    if is_train:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
