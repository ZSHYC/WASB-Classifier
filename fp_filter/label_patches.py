"""
交互式标注工具：读取 manifest.csv，逐个显示 patch，通过键盘输入标签（1=球，0=非球），并实时保存。
极大提高手动标注效率，无需在 Excel 和图片浏览器之间切换。

使用示例（在 src 目录下执行）：
python ../fp_filter/label_patches.py ^
    --manifest outputs/patches_match1_clip1/manifest.csv
"""

import os
import os.path as osp
import argparse
import pandas as pd
import cv2
import numpy as np

def label_patches(manifest_path, patch_size_display=256):
    """
    manifest_path: manifest.csv 的路径
    patch_size_display: 显示图片时的放大尺寸（原始 32x32 太小，建议放大查看）
    """
    if not osp.isfile(manifest_path):
        print(f"错误：文件不存在 {manifest_path}")
        return

    # 读取 CSV
    print(f"正在读取: {manifest_path}")
    df = pd.read_csv(manifest_path)

    # 确保 label 列存在
    if "label" not in df.columns:
        df["label"] = "" # 初始化为空字符串或 NaN

    manifest_dir = osp.dirname(osp.abspath(manifest_path))
    
    # 统计进度
    total = len(df)
    
    # 找到第一个未标注的索引
    # 判断标准：label 为 NaN 或 空字符串
    unlabeled_mask = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
    unlabeled_indices = df[unlabeled_mask].index.tolist()
    
    print(f"总样本数: {total}")
    print(f"剩余待标注: {len(unlabeled_indices)}")
    print("-" * 40)
    print("操作说明:")
    print("  [1] : 标记为 球 (TP)")
    print("  [0] : 标记为 非球 (FP)")
    print("  [Space] : 跳过当前")
    print("  [B] : 返回上一张 (Back)")
    print("  [Q] : 保存并退出")
    print("-" * 40)

    current_idx_ptr = 0 # 指向 unlabeled_indices 的指针
    history_stack = [] # 记录访问过的 index，用于回退

    while True:
        # 如果所有都标完了
        if current_idx_ptr >= len(unlabeled_indices):
            print("所有图片已标注完成！")
            break
            
        idx = unlabeled_indices[current_idx_ptr]
        row = df.iloc[idx]
        
        # 构造图片路径
        patch_rel_path = row["patch_path"]
        patch_full_path = patch_rel_path
        if not osp.isabs(patch_full_path):
            patch_full_path = osp.join(manifest_dir, patch_rel_path)
            
        if not osp.isfile(patch_full_path):
            print(f"[警告] 图片不存在，跳过: {patch_full_path}")
            current_idx_ptr += 1
            continue
            
        img = cv2.imread(patch_full_path)
        if img is None:
            print(f"[警告] 图片无法读取，跳过: {patch_full_path}")
            current_idx_ptr += 1
            continue

        # 放大图片以便查看
        img_display = cv2.resize(img, (patch_size_display, patch_size_display), interpolation=cv2.INTER_NEAREST)
        
        # 在图片上绘制信息
        info_text = f"ID: {current_idx_ptr+1}/{len(unlabeled_indices)} (Total: {idx})"
        cv2.putText(img_display, info_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_display, "1:Ball, 0:Bg, Q:Quit", (5, patch_size_display - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Labeling Tool", img_display)
        
        key = cv2.waitKey(0)

        # 按键处理
        if key == ord('q') or key == 27: # q or ESC
            print("正在保存并退出...")
            break
            
        elif key == ord('1'):
            df.at[idx, "label"] = 1
            print(f"[{idx}] marked as BALL (1)")
            history_stack.append(current_idx_ptr)
            current_idx_ptr += 1
            
        elif key == ord('0'):
            df.at[idx, "label"] = 0
            print(f"[{idx}] marked as BACKGROUND (0)")
            history_stack.append(current_idx_ptr)
            current_idx_ptr += 1
            
        elif key == ord(' '): # Space to skip
            print(f"[{idx}] Skipped")
            history_stack.append(current_idx_ptr)
            current_idx_ptr += 1
            
        elif key == ord('b') or key == ord('B'): # Back
            if len(history_stack) > 0:
                prev_ptr = history_stack.pop()
                current_idx_ptr = prev_ptr
                print(f"返回上一张 [{unlabeled_indices[current_idx_ptr]}]")
            else:
                print("已经是第一张了，无法回退。")
        else:
            print("无效按键，请按 1, 0, Q, Space 或 B")

        # 每 10 次操作自动保存一次，防止崩溃丢失
        if len(history_stack) % 10 == 0:
             df.to_csv(manifest_path, index=False)

    # 最终保存
    df.to_csv(manifest_path, index=False)
    print(f"保存成功: {manifest_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP 过滤二分类：交互式标注工具")
    parser.add_argument("--manifest", required=True, help="由 extract_patches.py 生成的 manifest.csv 路径")
    parser.add_argument("--size", type=int, default=256, help="显示窗口中的图片大小（像素）")
    
    args = parser.parse_args()
    
    label_patches(args.manifest, args.size)
