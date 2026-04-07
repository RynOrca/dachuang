# 基于扩散模型的文本图像超分辨率

> 超越 Final2x 的扩散模型文本图像增强方案

本项目是一个基于扩散模型的文本图像超分辨率（Text Image Super-Resolution）系统，专门针对文本图像进行优化，在整体增强的前提下**优先保证文字区域的边缘、笔画与可读性**。与传统的插值方法（如双三次插值）和现有工具（如 Final2x）相比，本方案利用扩散模型生成更清晰、更自然的文本细节，显著提升 OCR 准确率和视觉质量。

## ✨ 主要特性

- **扩散模型驱动**: 使用条件扩散模型进行图像超分，生成高质量细节
- **文本优先优化**: 在整体增强的基础上，特别优化文字区域的可读性
- **灵活的推理预设**: 提供 `fast`、`balanced`、`best` 以及专为文本设计的 `text-*` 预设
- **完整的评估体系**: 支持 PSNR、SSIM、LPIPS 等图像质量指标，以及 OCR CER/WER 等文本可读性指标
- **易用的统一入口**: `run_all.py` 脚本提供训练、推理、评估、报告生成等一站式功能
- **多 GPU 训练支持**: 支持 DDP 分布式训练，充分利用服务器硬件资源
- **可视化报告**: 自动生成 HTML 报告，包含对比画廊和详细指标分析

## 🆚 与 Final2x 对比

| 特性 | 本项目 | Final2x |
|------|--------|---------|
| 核心算法 | 扩散模型（生成式） | 传统超分模型（如 Real-ESRGAN） |
| 文本优化 | 专门针对文字区域优化 | 通用图像增强 |
| 细节生成 | 生成更自然的细节和纹理 | 可能产生伪影或过度平滑 |
| 可读性提升 | 显著改善文字边缘和笔画连续性 | 有限改进 |
| 评估指标 | 同时关注图像质量和 OCR 准确率 | 主要关注视觉质量 |
| 灵活性 | 多档预设，可平衡速度与质量 | 固定模型和参数 |

## 🚀 先看结论：你到底该怎么用这个项目

如果你只关心「输入一张模糊图，输出一张清晰图」，只需要记住这一条命令：

```bash
python run_all.py enhance \
  -i inputs/your_blur.png \
  -o outputs/your_result \
  --preset text-balanced
```

输出图会在 `outputs/your_result` 下，这就是最终可交付给用户的清晰图，不是中间过程图。

## ✅ 怎么判断“效果好不好”（用户视角）

建议按这 3 层判断：

1. **主观可读性（最重要）**
   - 看文字边缘是否更完整、断笔是否减少、噪点是否不过度。
   - 这是最终用户最关心的指标。

2. **OCR 指标（业务可落地）**
   - 看 `CER`、`WER`、`Accuracy`。
   - 目标是：`CER/WER` 越低越好，`Accuracy` 越高越好。

3. **图像指标（辅助）**
   - 看 `PSNR/SSIM/LPIPS`。
   - 目标是：`PSNR/SSIM` 越高越好，`LPIPS` 越低越好。

> 实战建议：如果 OCR 指标明显更好，同时主观可读性也更好，就可以认定“比 Final2x 更适合文本增强”。

---
大创成员：输入这个激活环境

```bash
chmod +x scripts/setup_realesrgan_env.sh
source scripts/setup_realesrgan_env.sh
```
##### WEEK_1
```bash
# Step 1: 从桌面源照片生成 GT 高清裁剪图
python tools/setup_eval_gt.py

# Step 2: 编辑 eval_labels/labels.csv，填写每张图的 gt_text（真实文字内容）

# Step 3: 一键跑通全流程（precheck → batch → OCR → report）
python scripts/run_week1_eval.py
```
回滚方案：
```bash
# 删除 3 个新文件
del tools\setup_eval_gt.py scripts\run_week1_eval.py eval_labels\labels.csv
# run_all.py 的改动仅是新增函数和参数，不影响任何已有功能
```


##### WEEK_2

```bash
# Step 1: 放置你的扩散模型 .pth 文件到 model/ 目录
#    （这是当前唯一硬阻塞项）

# Step 2: 用 prod-safe 预设测试单张图片
python run_all.py enhance -i eval_inputs/eval_001.png -o outputs/test \
  --preset prod-safe --model_path model/<你的模型>.pth --save_comparison
```
回滚方案：
```bash
git checkout -- inference_diffusion.py run_all.py
```

##### WEEK_3

重新训练模型：
```bash
python train_diffusion.py --hr_dir dataset/HR --degradation text_realistic \
  --val_split 0.08 --val_every 5 --batch_size 8 --epochs 1000 \
  --experiment_name diffusion_week3_textreal \
  --save_dir model

# 多卡训练：

export CUDA_VISIBLE_DEVICES=0,1,2,4,5,7

torchrun --nproc_per_node=2 train_diffusion.py --hr_dir dataset/HR --degradation text_realistic \
  --val_split 0.08 --val_every 5 --batch_size 8 --epochs 1000 \
  --experiment_name diffusion_week3_textreal \
  --save_dir model

  ```


---


## 📦 安装

### 环境要求

- Python 3.8+
- PyTorch 1.7+（支持 CUDA）

### 安装命令

```bash
conda create -n text-enhance python=3.9
conda activate text-enhance

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt

# 可选：OCR 评估
pip install paddleocr

# 可选：LPIPS 评估
pip install lpips
```

## 🧭 一条主线：训练 → 选模型 → 测试 → 产出清晰图

### 1) 训练（单卡示例）

```bash
python train_diffusion.py \
  --cond_mode concat \
  --batch_size 4 \
  --epochs 200 \
  --scale 4 \
  --hr_size 256 \
  --train_size 256 \
  --lr 1e-4 \
  --lambda_seg 0.2 \
  --num_workers 4 \
  --hr_dir dataset_triplet/train/HR \
  --lr_dir dataset_triplet/train/LR \
  --mask_dir dataset_triplet/train/masks \
  --save_dir model \
  --experiment_name diffusion_textzoom_bs8 \
  --save_best \
  --save_every 5
```

训练输出模型命名规则：

- 最新权重：`model/<experiment_name>_latest.pth`
- 最优权重：`model/<experiment_name>_best.pth`

例如上面的命令会得到：

- `model/diffusion_textzoom_bs8_latest.pth`
- `model/diffusion_textzoom_bs8_best.pth`

### 2) 用你训练好的模型直接清晰化图片（核心命令）

```bash
python run_all.py enhance \
  -i inputs \
  -o outputs/enhanced \
  --model_path model/diffusion_textzoom_bs8_best.pth \
  --preset text-balanced
```

> 这一步产出的就是你最终要给用户看的结果图。

### 3) 和 Final2x（Real-ESRGAN）做同场对比

```bash
python run_all.py batch \
  --input_dir eval_inputs \
  --output_dir eval_outputs/cmp_final2x \
  --methods bicubic,realesrgan,diffusion \
  --model_path model/diffusion_textzoom_bs8_best.pth \
  --preset text-balanced \
  --gt_dir eval_gt \
  --lpips
```

输出目录说明：

- `eval_outputs/cmp_final2x/realesrgan/`：Final2x 基线输出
- `eval_outputs/cmp_final2x/diffusion/`：本项目输出
- `eval_outputs/cmp_final2x/comparisons/`：拼图对比图
- `eval_outputs/cmp_final2x/metrics.csv`：PSNR/SSIM/LPIPS
- `eval_outputs/cmp_final2x/summary.json`：整体汇总

### 4) OCR 评测（判断“可读性是否真的更好”）

```bash
python run_all.py ocr-eval \
  --pred_dir eval_outputs/cmp_final2x/diffusion \
  --gt_csv eval_labels/labels.csv \
  --image_col image \
  --text_col text \
  --ocr_backend paddleocr \
  --lang ch \
  --device gpu \
  --output_csv ocr_metrics_diffusion.csv \
  --output_json ocr_metrics_diffusion.json
```

再将 `--pred_dir` 改成 `eval_outputs/cmp_final2x/realesrgan` 再跑一次，就能直接比较你和 Final2x 的 OCR 指标。

### 5) 一键完整评估（图像指标 + OCR + 报告）

```bash
python run_all.py full-eval \
  --input_dir eval_inputs \
  --output_dir eval_outputs/full_evaluation \
  --gt_dir eval_gt \
  --gt_csv eval_labels/labels.csv \
  --methods bicubic,diffusion \
  --model_path model/diffusion_textzoom_bs8_best.pth \
  --lpips \
  --report_html full_eval_report.html
```

说明：

- `full-eval` 会默认把 OCR 输入目录设为 `<output_dir>/diffusion`。
- 若你想评估其他目录，可显式加 `--pred_dir`。

## 📁 数据文件约定（避免“找不到 gt_csv/gt_dir”）

- `eval_inputs/`：待增强输入图
- `eval_gt/`：与输入同名的 GT 清晰图（用于 PSNR/SSIM/LPIPS）
- `eval_labels/labels.csv`：OCR 标签 CSV（至少包含 `image` 和 `text` 两列）

示例 CSV：

```csv
image,text
eval_001.png,欢迎使用文本增强
eval_002.png,发票号码123456
```

## 🏗️ 高级功能

### 模型注册表

```bash
# 列出可用模型
python run_all.py model-registry --action list

# 验证模型完整性
python run_all.py model-registry --action verify --model text-priority

# 下载模型
python run_all.py model-registry --action download --model text-priority
```

### 预设管理

```bash
# 列出所有预设
python run_all.py preset --action list

# 创建自定义预设
python run_all.py preset --action set \
  --name my-text \
  --values_json '{"steps":180,"min_side":352,"edge_sharpen_strength":0.4}'

# 删除预设
python run_all.py preset --action delete --name my-text
```

### 任务队列

创建 `tasks.json`：
```json
{
  "tasks": [
    {
      "name": "compare-fast",
      "argv": ["compare", "--input_dir", "eval_inputs", "--output_dir", "eval_outputs/cmp_fast", "--preset", "text-fast"]
    },
    {
      "name": "report-fast",
      "argv": ["report", "--output_dir", "eval_outputs/cmp_fast", "--summary_json", "compare_summary.json"]
    }
  ]
}
```

执行任务队列：
```bash
python run_all.py queue \
  --queue_json tasks.json \
  --history_json queue_history.json \
  --stop_on_error
```

### GUI 界面

```bash
python run_all.py gui
```

启动桌面 GUI，提供可视化操作界面。

## ❓ 常见问题

### Q1: CUDA 内存不足（OOM）

**解决方法**：
1. 减小批大小：`--batch_size 4 -> 2`
2. 减小训练尺寸：`--train_size 256 -> 128`
3. 减小推理尺寸：`--target_min_side 384 -> 256`
4. 减少采样步数：`--timesteps 200 -> 120`

### Q2: 训练效果不明显

**建议**：
1. 增加训练轮数：`--epochs 200 -> 400`
2. 增加采样步数：`--timesteps 120 -> 200`
3. 检查数据质量，确保 HR/LR 对应正确
4. 尝试调整学习率：`--lr 1e-4 -> 5e-5`

### Q3: 如何选择最佳模型

**优先级**：
1. 先看 OCR：`CER/WER` 更低、`Accuracy` 更高者优先
2. 再看主观可读性：文字边缘、断笔、伪影
3. 最后参考 PSNR/SSIM/LPIPS
4. 优先使用 `*_best.pth` 作为线上候选模型

### Q4: OCR 评估失败

**检查步骤**：
1. 确认安装了 PaddleOCR：`pip show paddleocr`
2. 确认 PaddlePaddle GPU 版本与 CUDA 匹配
3. 检查 PP-OCRv5 模型文件是否存在
4. 确认 `--gt_csv` 路径存在，且包含 `image` 和 `text` 列
5. 确认 `--pred_dir` 与当前评测输出目录一致（例如 `eval_outputs/full_evaluation/diffusion`）

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本项目。

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交 Pull Request

## 🙏 致谢

- 感谢 TextZoom 数据集提供方
- 感谢 Real-ESRGAN 项目提供的退化方法
- 感谢 PaddleOCR 团队提供的 OCR 工具

---

**如有问题，请查看 [Issues](https://github.com/your-username/dachuang-dachuang-text-enhancement/issues) 或提交新 Issue。**