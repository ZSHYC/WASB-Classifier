# FP 过滤二分类：使用说明

通过“先提取 patch → 人工标注 → 训练二分类模型”的方式，筛除检测结果中的假阳性（FP），保留真阳性（TP）。

---
 
## 第一步：从检测结果提取 Patch

在完成 WASB 预测并得到 `match1_clip1_predictions.csv` 后，用脚本从**原始数据集**中，以每个 `visibility=1` 的检测点 (x, y) 为中心截取小图块（patch），并生成供标注用的 manifest。

### 1. 运行提取脚本

在 **`src`** 目录下执行（以下路径均相对 `src`）：

```powershell
cd C:\Users\mayuchao\Desktop\WASB-SBDT\src

python ../fp_filter/extract_patches.py ^
  "outputs/main/2026-02-05_11-10-50/match1_clip1_predictions.csv" ^
  --dataset-root ../datasets/tennis_predict ^
  --output-dir outputs/patches_match1_clip1 ^
  --patch-size 96
```

- **第一个参数**：预测结果 CSV 的路径。
- **`--dataset-root`**：原始数据集根目录（与 `configs/dataset/tennis_predict.yaml` 中的 `root_dir` 对应，即帧所在根目录）。
- **`--output-dir`**：输出目录，将在此生成所有 patch 图片和 `manifest.csv`。
- **`--patch-size`**：patch 边长（像素），默认 32（常见小目标尺寸）。若第一步用其他尺寸，第二步训练时需用 `--patch-size` 保持一致。

脚本会从 CSV 文件名解析 `match` 和 `clip`（如 `match1_clip1_predictions.csv` → match1, clip1），帧路径为：  
`dataset_root / match / clip / 文件名`。  
若目录结构不同，可用 `--match`、`--clip` 显式指定。

### 2. 输出内容

- **`output_dir/*.png`**：以 `(x,y)` 为中心截取的 patch 图。
- **`output_dir/manifest.csv`**：表格列包括  
  `patch_id, source_file, x, y, match, clip, patch_path, score, label`。  
  其中 **`label` 列为空**，供你标注。

### 3. 标注

#### 方式 A：使用交互式标注脚本（推荐）

使用 `fp_filter/label_patches.py` 工具，逐张查看图片并按键标注，效率极高。

在 **项目根目录** 下执行：

```powershell
python fp_filter/label_patches.py --manifest fp_filter/patch_outputs/patches_match1_clip1/manifest.csv --size 512
```

若需要从第 100 张图片开始标注（续接之前的进度）：

```powershell
python fp_filter/label_patches.py --manifest fp_filter/patch_outputs/patches_match1_clip1/manifest.csv --size 512 --start-from 100
```

**操作方法**：
- **按 `1`**：标记为 **球 (TP)**，自动保存并跳转下一张。
- **按 `0`**：标记为 **非球 (FP)**，自动保存并跳转下一张。
- **按 `B`**：回退到上一张（如果标错了）。
- **按 `Q`**：保存并退出。
- **UI 显示**：界面上会实时显示当前图片的标记状态（None / Ball(1) / Background(0)）。

#### 方式 B：手动编辑 CSV

- 打开 `manifest.csv`。
- 结合图片文件夹查看对应图片。
- 对每一行在 **`label`** 列填写：`1` (球) 或 `0` (非球)。
- 保存为 UTF-8（如 Excel 另存为 CSV UTF-8）。

---

## 第二步：训练二分类模型

用标注好的 manifest 训练一个小型 CNN，输入为 patch 图像，输出二类：球(1) / 非球(0)。

### 1. 准备

- 已完成第一步，且已在 `manifest.csv` 中填好 **`label`**（0 或 1）。
- `manifest.csv` 中的 `patch_path` 列指向的图片文件存在且可读。

### 2. 训练命令（在 `src` 下执行）

```powershell
cd C:\Users\mayuchao\Desktop\WASB-SBDT\src

python ../fp_filter/train_fp_filter.py ^
  --manifest outputs/patches_match1_clip1/manifest.csv ^
  --out-dir outputs/fp_filter ^
  --val-ratio 0.2 ^
  --epochs 50 ^
  --batch-size 64 ^
  --lr 1e-3 ^
  --patch-size 96
```

- **`--manifest`**：已标注 label 的 manifest 路径。
- **`--out-dir`**：保存 checkpoint 和 `history.json` 的目录。
- **`--patch-size`**：与第一步使用的 patch 尺寸一致（如 96）。

### 3. 输出

- **`outputs/fp_filter/best.pth`**：验证集准确率最高的模型，可用于后续推理筛 FP。
- **`outputs/fp_filter/last.pth`**：最后一轮模型。
- **`outputs/fp_filter/history.json`**：每轮 train/val 的 loss 与 accuracy。

---

## 模型与数据说明

- **Patch 尺寸**：默认 96×96。可在第一步、第二步用同一 `--patch-size` 修改。
- **二分类网络**：`tools/fp_filter/model.py` 中的轻量 CNN（若干 Conv+BN+ReLU+Pool，最后全连接 2 类）。如需更大/更小模型，可在此文件内修改或替换。
- **数据增强**：训练时仅做随机水平翻转；如需更多增强，可修改 `fp_filter/dataset.py` 中的 `get_default_transform`。

---

## 第三步：推理与过滤

使用训练好的 `best.pth` 对新的检测结果 CSV 进行过滤。

### 1. 运行推理脚本

在 **`src`** 目录下执行：

```powershell
cd C:\Users\mayuchao\Desktop\WASB-SBDT\src

python ../fp_filter/inference.py ^
  --csv "outputs/main/2026-02-05_11-10-50/match1_clip1_predictions.csv" ^
  --dataset-root ../datasets/tennis_predict ^
  --model "outputs/fp_filter/best.pth" ^
  --output "outputs/main/2026-02-05_11-10-50/match1_clip1_predictions_filtered.csv" ^
  --threshold 0.5
```

- **`--csv`**：待过滤的检测结果 CSV。
- **`--dataset-root`**：原始数据集根目录。
- **`--model`**：第二步训练好的模型路径。
- **`--output`**：输出的新 CSV 文件路径。
- **`--threshold`**：分类阈值（默认 0.5），低于此值的检测点将被视为 FP（`visibility` 置为 0）。

脚本会生成一个新的 CSV 文件，其中包含过滤后的结果，并增加 `fp_score` 列记录二分类模型的打分。
