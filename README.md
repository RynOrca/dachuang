# 文本图像超分（扩散模型）

本仓库当前支持一条可在本地运行的完整流程：

1. 从 TextZoom 的 LMDB 提取 HR 图片
2. 生成三联数据（`HR/LR/masks`，其中 LR 由 Real-ESRGAN 风格退化生成）
3. 训练扩散模型
4. 推理并导出结果
5. 使用统一脚本对比 `input_x4_nearest / bicubic / diffusion`

---

>`input_x4_nearest`：把输入图像直接做 最近邻插值 放大到 4 倍（x4）。不会新增细节，只是把像素块放大
>`bicubic`：用 双三次插值（Bicubic interpolation） 放大图像

input_x4_nearest：粗糙放大参考
bicubic：平滑插值参考
diffusion：模型恢复结果
要判断模型有没有价值，就看 diffusion 是否在文字边缘、断笔、可读性上明显优于前两者。

目前diffusion 是核心模型；input_x4_nearest 和 bicubic 只是对比基线。
未完全实现：严格定义的固定倍率超分（如“输入必定 x4 输出”）

---

> 建议环境：Windows + Conda（`pytorch` 环境），RTX 4060 Laptop（8GB 显存）可跑通本流程。

---

## 1. 环境准备

```powershell
conda activate pytorch
python -m pip install -r requirements.txt
```

---

## 2. 从 TextZoom 提取 HR 图像到 `dataset/HR`

TextZoom 目录示例：

- `TextZoom/train1`（`data.mdb`, `lock.mdb`）
- `TextZoom/train2`（`data.mdb`, `lock.mdb`）
- `TextZoom/test/...`（用于后续评估，可暂不参与训练）

执行命令：

```powershell
python .\tools\extract_lmdb_images_generic.py --lmdb_dir .\TextZoom\train1 --out_dir .\dataset\HR --prefix train1 --only_hr
python .\tools\extract_lmdb_images_generic.py --lmdb_dir .\TextZoom\train2 --out_dir .\dataset\HR --prefix train2 --only_hr
```

参数说明：

- `--lmdb_dir`：输入 LMDB 目录
- `--out_dir`：输出图片目录
- `--prefix`：输出文件名前缀，避免重名
- `--only_hr`：仅提取包含 HR 标记的图像 key

检查是否提取成功：

```powershell
Test-Path .\dataset\HR
(Get-ChildItem .\dataset\HR -File -Recurse | Measure-Object).Count
```

---

## 3. 生成三联数据（HR/LR/masks）

脚本：`tools/make_triplet_from_hr.py`

### 3.1 无 mask（先跑通）

```powershell
python .\tools\make_triplet_from_hr.py `
	--hr_dir .\dataset\HR `
	--out_root .\dataset_triplet\train `
	--scale 4
```

### 3.2 有 mask（同名）

```powershell
python .\tools\make_triplet_from_hr.py `
	--hr_dir .\dataset\HR `
	--mask_dir .\dataset\masks `
	--out_root .\dataset_triplet\train `
	--scale 4
```

输出结构：

```text
dataset_triplet/
	train/
		HR/
		LR/
		masks/   # 仅在传入 --mask_dir 时生成
```

关键参数说明：

- `--scale 4`：按 x4 关系生成 LR（尺寸约为 HR 的 1/4）
- `--mask_dir`：可选，复制同名 mask 形成三联

---

## 4. 本地训练（1-3 小时模板）

命令：

```powershell
python .\train_diffusion.py `
	--cond_mode concat `
	--batch_size 4 `
	--epochs 30 `
	--hr_size 128 `
	--train_size 128 `
	--lr 1e-4 `
	--lambda_seg 0 `
	--num_workers 2 `
	--hr_dir .\dataset_triplet\train\HR `
	--lr_dir .\dataset_triplet\train\LR `
	--mask_dir .\dataset_triplet\train\masks `
	--save_dir .\experiments `
	--experiment_name diffusion_local_1to3h `
	--save_every 2
```

参数说明：

- `--cond_mode concat`：条件输入模式（推荐起步使用）
- `--batch_size`：显存敏感参数，OOM 时优先减小
- `--hr_size / --train_size`：训练分辨率
- `--lambda_seg 0`：无可靠 mask 时建议先设 0
- `--hr_dir / --lr_dir / --mask_dir`：三联数据目录
- `--experiment_name`：输出模型文件名前缀

训练产物示例：

- `experiments/diffusion_local_1to3h_latest.pth`

---

## 5. 推理

```powershell
python .\inference_diffusion.py `
	-i .\eval_inputs `
	-o .\eval_outputs\diffusion_local_1to3h `
	--model_path .\experiments\diffusion_local_1to3h_latest.pth `
	--timesteps 120
```

参数说明：

- `-i`：输入图片目录
- `-o`：输出目录
- `--model_path`：训练得到的 checkpoint
- `--timesteps`：采样步数（更高通常更慢但可能更好）

---

## 6. 统一对比评测（输入放大 vs bicubic vs diffusion）

```powershell
python .\tools\evaluate_text_models.py `
	--input_dir .\eval_inputs `
	--output_dir .\eval_outputs\cmp_local `
	--methods bicubic,diffusion `
	--outscale 4 `
	--diffusion_model_path .\experiments\diffusion_local_1to3h_latest.pth `
	--diffusion_steps 80 `
	--diffusion_min_side 256 `
	--diffusion_fallback_min_side 192
```

打开对比图目录：

```powershell
Start-Process .\eval_outputs\cmp_local\comparisons
```

说明：

- `--methods bicubic,diffusion`：只跑双三次 + 扩散，不依赖 `basicsr`
- `--diffusion_min_side`：扩散推理尺度，过大可能 OOM
- `--diffusion_fallback_min_side`：OOM 自动回退尺度

---

## 7. 常见问题

### Q1: `ModuleNotFoundError: basicsr`

- 若只跑 `bicubic,diffusion`，当前脚本已支持不安装 `basicsr`
- 若要加 `realesrgan` 方法，请安装：

```powershell
python -m pip install basicsr
```

### Q2: CUDA OOM

优先按顺序调整：

1. 训练：`--batch_size 4 -> 2`
2. 训练：`--train_size 128 -> 96`
3. 评测：`--diffusion_min_side 256 -> 192`
4. 评测：`--diffusion_steps 80 -> 60`

### Q3: 看不出效果差异

- 提升训练轮数（如 `epochs 80+`）
- 评测提高采样步数（如 `diffusion_steps 120~200`）
- 对比时重点看文字边缘、断笔、重影、可读性

---

## 8. 最短复现路径

```powershell
conda activate pytorch
python -m pip install -r requirements.txt
python .\tools\extract_lmdb_images_generic.py --lmdb_dir .\TextZoom\train1 --out_dir .\dataset\HR --prefix train1 --only_hr
python .\tools\extract_lmdb_images_generic.py --lmdb_dir .\TextZoom\train2 --out_dir .\dataset\HR --prefix train2 --only_hr
python .\tools\make_triplet_from_hr.py --hr_dir .\dataset\HR --out_root .\dataset_triplet\train --scale 4
python .\train_diffusion.py --cond_mode concat --batch_size 4 --epochs 30 --hr_size 128 --train_size 128 --lr 1e-4 --lambda_seg 0 --num_workers 2 --hr_dir .\dataset_triplet\train\HR --lr_dir .\dataset_triplet\train\LR --save_dir .\experiments --experiment_name diffusion_local_1to3h --save_every 2
python .\tools\evaluate_text_models.py --input_dir .\eval_inputs --output_dir .\eval_outputs\cmp_local --methods bicubic,diffusion --outscale 4 --diffusion_model_path .\experiments\diffusion_local_1to3h_latest.pth --diffusion_steps 80 --diffusion_min_side 256 --diffusion_fallback_min_side 192
```

---

## 9. 服务器推荐命令

> 说明：`train_diffusion.py` 已支持 `--ddp`，请使用 `torchrun` 启动多卡同步训练。

### 9.1 6卡 DDP 训练（质量优先）

```bash
torchrun --nproc_per_node=6 train_diffusion.py \
	--ddp \
	--dist_backend nccl \
	--cond_mode concat \
	--batch_size 16 \
	--epochs 200 \
	--hr_size 256 \
	--train_size 256 \
	--lr 8e-5 \
	--lambda_seg 0.2 \
	--num_workers 8 \
	--hr_dir ./dataset_triplet/train/HR \
	--lr_dir ./dataset_triplet/train/LR \
	--mask_dir ./dataset_triplet/train/masks \
	--save_dir ./experiments \
	--experiment_name diffusion_ddp_6x4090 \
	--save_every 5 \
	--archive_every 20
```

### 9.2 6卡 DDP 续训

```bash
torchrun --nproc_per_node=6 train_diffusion.py \
	--ddp \
	--dist_backend nccl \
	--resume \
	--cond_mode concat \
	--batch_size 16 \
	--epochs 200 \
	--hr_size 256 \
	--train_size 256 \
	--lr 8e-5 \
	--lambda_seg 0.2 \
	--num_workers 8 \
	--hr_dir ./dataset_triplet/train/HR \
	--lr_dir ./dataset_triplet/train/LR \
	--mask_dir ./dataset_triplet/train/masks \
	--save_dir ./experiments \
	--experiment_name diffusion_ddp_6x4090 \
	--save_every 5 \
	--archive_every 20
```

---

### 9.3 服务器测试（单模型推理）

```bash
CUDA_VISIBLE_DEVICES=0 python inference_diffusion.py \
	-i ./eval_inputs \
	-o ./eval_outputs/diffusion_ddp_6x4090 \
	--model_path ./experiments/diffusion_ddp_6x4090_latest.pth \
	--timesteps 200 \
	--target_min_side 384
```

- 质量优先：`timesteps=200~300`
- 速度优先：`timesteps=80~120`

### 9.4 服务器统一评测（推荐）

```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/evaluate_text_models.py \
	--input_dir ./eval_inputs \
	--output_dir ./eval_outputs/cmp_ddp_6x4090 \
	--methods bicubic,diffusion \
	--outscale 4 \
	--diffusion_model_path ./experiments/diffusion_ddp_6x4090_latest.pth \
	--diffusion_steps 160 \
	--diffusion_min_side 384 \
	--diffusion_fallback_min_side 256
```

如果有 GT，可加：

```bash
--gt_dir ./eval_gt --metrics_csv metrics_ddp_6x4090.csv
```

---

### 9.5 如何从当前实验选最佳

优先级建议：

1. `metrics.csv`（若有 GT）：按 `SSIM`、`PSNR` 选 Top-2 checkpoint
2. 人工目检：重点看文字边缘、断笔、重影、可读性
3. 最终保留：`best_latest.pth` + 对应训练命令（可复现）
