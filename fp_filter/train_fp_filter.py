"""
第二步：使用标注好的 manifest（含 label 列）训练二分类模型，用于筛除 FP。

使用示例（在 fp_filter 目录下执行）：cd fp_filter
python train_fp_filter.py ^
  --manifest patch_outputs/patches_match1_clip1/manifest.csv ^
  --out-dir patch_outputs/fp_filter ^
  --val-ratio 0.2 ^
  --epochs 50
"""
import os
import sys
import os.path as osp
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 保证从项目根或 src 运行都能找到 fp_filter（在 import 前执行）
_src = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from fp_filter.dataset import PatchDataset, get_default_transform
from fp_filter.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="训练 FP 过滤二分类模型")
    parser.add_argument("--manifest", "-m", required=True, help="已标注 label 的 manifest.csv 路径")
    parser.add_argument("--out-dir", "-o", default="./outputs/fp_filter", help="保存 checkpoint 与日志的目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--patch-size", type=int, default=96, help="与第一步提取的 patch 尺寸一致，仅用于 transform")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = get_default_transform(is_train=True, patch_size=args.patch_size)
    transform_val = get_default_transform(is_train=False, patch_size=args.patch_size)

    full_dataset = PatchDataset(args.manifest, transform=None, target_one_hot=False)
    n = len(full_dataset)
    if n == 0:
        raise ValueError("manifest 中没有已标注的样本，请先在 manifest 的 label 列填 1(球) 或 0(非球)")

    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    train_dataset = PatchDataset(args.manifest, transform=transform_train, target_one_hot=False)
    val_dataset = PatchDataset(args.manifest, transform=transform_val, target_one_hot=False)
    train_sub = torch.utils.data.Subset(train_dataset, train_idx.tolist())
    val_sub = torch.utils.data.Subset(val_dataset, val_idx.tolist())

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Some PyTorch versions don't accept the `verbose` kwarg here; omit it for compatibility.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = osp.join(args.out_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  保存最佳模型: {ckpt_path}")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(), "args": vars(args)},
               osp.join(args.out_dir, "last.pth"))
    with open(osp.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
