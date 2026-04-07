# Agent 恢复执行包（跨会话粘贴即用）

> 用途：当助手"失忆"或跨会话后，把本文件内容贴给助手，即可迅速恢复到正确步骤。

## 1) 当前项目目标（固定）
- 项目：`dachuang-dachuang-text-enhancement`
- 目标：商业级文本超分交付，**达到 Final2x/RealESRGAN 级别的清晰度**，不允许"加噪变糊"。
- 策略：`realesrgan` 保底 + `diffusion` 文本增强（DDIM 稳像） + OCR 闭环。

## 2) 当前执行阶段（每周更新）
- 当前周次：`WEEK_3`
- 当前状态：`IN_PROGRESS — 训练链路工业化改造完成（退化策略+验证日志+多指标选优），待重新训练验证`
- 当前主任务：`训练链路工业化（数据一致性与恢复能力）`
- 下一主任务：`重新训练模型 + 验证质量是否达到 Final2x 水平`

### 历史代码改动汇总

| 会话 | 改动 | 文件 | 说明 |
|------|------|------|------|
| S1 | 新增 | `tools/setup_eval_gt.py` | 从 manifest.csv 源图自动裁剪生成 eval_gt/ |
| S1 | 新增 | `scripts/run_week1_eval.py` | WEEK_1 一键验收脚本（6 步流水线） |
| S1 | 新增 | `eval_labels/labels.csv` | OCR 标注模板 |
| S1 | 增强 | `run_all.py` precheck | labels_csv 格式校验 + strict_eval_set 交叉检查 |
| **S2** | **核心改造** | **`inference_diffusion.py`** | **DDIM 确定性采样、条件初始化、后处理双边滤波去噪、智能降级、质量评分** |
| **S2** | **核心改造** | **`run_all.py`** | **新增生产预设 prod-safe/prod-quality，修复布尔覆盖 bug** |
| **S3** | **核心改造** | **`dataloader.py`** | **新增 TextDegradationPipeline 类（运动模糊/JPEG压缩/局部遮挡/扫描噪声/散焦模糊），5 级预设 none/light/medium/heavy/text_realistic；TextSRDataset 集成 degradation 参数透传** |
| **S3** | **核心改造** | **`train_diffusion.py`** | **新增 compute_psnr_ssim / compute_composite_score / run_validation 函数；build_dataloader 支持 val_split 自动切分验证集；train() 每 val_every epoch 运行 PSNR/SSIM 验证并记录到 CSV；checkpoint 选优从纯 MSE 改为 composite_score（loss40% + PSNR35% + SSIM25%）；新增 --degradation/--val_split/--val_every CLI 参数** |

## 3) 验收项

### WEEK_1（数据闭环）
- [x] **评测集目录结构固化**：`eval_inputs/` ✅ / `eval_gt/` 📁(需本地执行 setup_eval_gt.py) / `eval_labels/labels.csv` ✅(模板已建)
- [ ] **报告含 OCR + 视觉指标**：代码管线就绪，需本地跑通后生成 `week1_report.html`
- [ ] **产出 TopN 失败样本列表**：代码支持就绪，待全流程复跑

### WEEK_2 前置（推理稳像 — 已完成）
- [x] **DDIM 确定性采样**：消除虎皮纹/随机失真，同一输入始终产生相同输出
- [x] **条件初始化**：从加噪输入图启动扩散（而非纯噪声），大幅减少伪影
- [x] **后处理去噪滤波器**：双边滤波在保留边缘的同时平滑纹理噪声
- [x] **智能降级策略**：检测到扩散输出退化时自动回退到 bicubic/realesrgan
- [x] **生产预设 `prod-safe` / `prod-quality`**：开箱即用的稳像配置

### WEEK_3（训练链路工业化 — 本次会话完成）
- [x] **任务1: 数据退化策略升级**：TextDegradationPipeline 实现 5 种文本专项退化（运动模糊/JPEG压缩/局部遮挡/扫描噪声/散焦模糊），5 级可配置预设，默认 `text_realistic`
- [x] **任务2: 训练日志升级**：每 val_every epoch 自动运行验证集 PSNR/SSIM 评估并写入 train_log.csv；summary_json 同步记录最佳验证指标
- [x] **任务3: Checkpoint 多指标综合选优**：composite_score = 0.4×loss_norm + 0.35×psnr_sigmoid + 0.25×ssim；不再仅依赖 MSE loss 选 best

### ⏳ 待执行（下一步）
- [ ] **重新训练模型**：使用新退化管线（`--degradation text_realistic`）+ 验证指标监控重新训练
- [ ] **质量验收**：用新训练的模型跑 WEEK_1 评估流水线，对比 Final2x 基线

### ⚠️ 用户本地操作清单（阻塞项）
```bash
# === 必须先完成的操作 ===

# 1. 放置扩散模型权重文件到 model/ 目录
#    （当前 model/ 目录不存在，需要你的训练 checkpoint）
#    例如: model/diffusion_textzoom_bs8_latest.pth 或 diffusion_fix_sanity_best.pth

# 2. 从源照片裁剪 GT 高清图（源图在桌面 phone_photos/）
python tools/setup_eval_gt.py

# 3. 编辑 eval_labels/labels.csv，填写每张图的 gt_text（真实文字内容）

# === 验证命令（按顺序执行）===

# 4a. 使用 prod-safe 预设单图测试（推荐首次使用）
python run_all.py enhance -i eval_inputs/eval_001.png -o outputs/test_prod_safe \
  --preset prod-safe --model_path <你的模型路径> --save_comparison

# 4b. 使用 prod-quality 预设测试（更高质量但更慢）
python run_all.py enhance -i eval_inputs/eval_001.png -o outputs/test_prod_quality \
  --preset prod-quality --model_path <你的模型路径> --save_comparison

# 5. 一键执行完整 WEEK_1 验收流水线（需要 GT 已填充）
python scripts/run_week1_eval.py --skip_gt_setup

# 6. 批量对比所有方法（含新的 DDIM 模式）
python run_all.py batch --input_dir eval_inputs --output_dir eval_outputs/cmp_ddim \
  --methods bicubic,realesrgan,diffusion,diffusion_realesrgan \
  --model_path <你的模型路径> --preset prod-quality --gt_dir eval_gt --lpips
```

## 4) 当前关键命令（可直接复用）

### 🔥 重新训练（WEEK_3 新增——推荐使用文本退化管线）
```bash
# 使用 text_realistic 退化 + 验证监控训练（推荐）
python train_diffusion.py --hr_dir dataset/HR --degradation text_realistic \
  --val_split 0.08 --val_every 5 --batch_size 8 --epochs 1000 \
  --experiment_name diffusion_week3_textreal \
  --save_dir model --save_every 10 --archive_every 100

# 轻度退化（适合数据量少时避免过拟合）
python train_diffusion.py --hr_dir dataset/HR --degradation light \
  --val_split 0.08 --val_every 5 --batch_size 8 --epochs 500

# 无退化（回退到旧行为，用于对比实验）
python train_diffusion.py --hr_dir dataset/HR --degradation none \
  --val_split 0 --epochs 500
```

### 🔥 生产预设一键使用（推荐）
```bash
# prod-safe: 快速稳定模式（100步，强去噪，智能降级兜底）
python run_all.py enhance -i your_image.png -o output_safe \
  --preset prod-safe --model_path <模型路径> --save_comparison

# prod-quality: 高质量模式（180步，适度去噪，保留更多细节）
python run_all.py enhance -i your_image.png -o output_quality \
  --preset prod-quality --model_path <模型路径> --save_comparison
```

### 手动精细调参（高级用法）
```bash
# 完全确定性 DDIM + 条件初始化 + 中等去噪
python inference_diffusion.py -i eval_inputs/ -o outputs/ddim_test \
  --model_path <模型路径> --use_ddim --ddim_eta 0.0 \
  --init_from_input --noise_strength 0.1 \
  --post_denoise_strength 0.3 --smart_fallback \
  --strict_color_lock --enhance_strength 0.9 \
  --timesteps 180 --save_comparison
```

### WEEK_1 一键验收
```bash
python scripts/run_week1_eval.py
```

### 批量对比（含混合方案）
```bash
python run_all.py batch \
  --input_dir eval_inputs \
  --output_dir eval_outputs/cmp_weekly \
  --methods bicubic,realesrgan,diffusion,diffusion_realesrgan \
  --model_path <你的模型路径> \
  --preset prod-quality \
  --gt_dir eval_gt \
  --lpips
```

## 5) 给助手的"恢复提示词"（直接复制）

```text
请读取并严格执行 docs/COMMERCIAL_EXECUTION_PLAN.md 与 docs/AGENT_RESUME_PACKET.md。
当前周次是 WEEK_3，当前主任务是"训练链路工业化（数据一致性与恢复能力）"，已完成退化策略升级+验证日志+多指标选优代码改造。
请先汇报当前仓库相对于这两个文档的进度差异（仅差异），然后立刻继续执行下一步代码改造，不要只给建议。
执行中每完成一项就更新 docs/AGENT_RESUME_PACKET.md 的当前阶段和验收勾选。
质量优先，所有改动都要可回滚且可验证。
```

## 6) 更新规则（非常重要）
- 每次完成一个里程碑，必须更新本文件第 2/3 节。
- 若策略改变，先改 `COMMERCIAL_EXECUTION_PLAN.md`，再改本文件。
- 本文件永远保持"下一次会话可接续"的状态。

## 7) 回滚信息

### 所有改动均为**向后兼容增强**（默认值不变，旧行为 100% 保留）：

| 文件 | 改动类型 | 回滚方式 |
|------|---------|---------|
| `dataloader.py` | 新增 TextDegradationPipeline 类 + TextSRDataset 扩展参数 | 删除第 8-151 行的 TextDegradationPipeline 类；TextSRDataset.__init__ 恢复旧签名（删 degradation/**deg_kwargs）；_make_lr 删第 5 步退化管线 |
| `train_diffusion.py` | 新增 3 个验证函数 + build_dataloader 扩展 + train() 验证循环 + CLI 参数 | 删除 compute_psnr_ssim/compute_composite_score/run_validation 函数；build_dataloader 恢复 4 返回值签名；train() 恢复纯 MSE 选优逻辑；删除 --degradation/--val_split/--val_every 参数 |
| `inference_diffusion.py` | 新增函数 + 扩展参数 | 删除新增函数即可回滚；原 DDPM 采样路径完全保留 |
| `run_all.py` | 新增参数透传 + 新预设 + 修复布尔覆盖 | 删除 PRESETS 中的 prod-safe/prod-quality 行；删除 add_shared_diffusion_args 中的新行；删除 handle_enhance 中的新透传 |

### 完整回滚命令：
```bash
git checkout -- dataloader.py train_diffusion.py inference_diffusion.py run_all.py
# 如非 git 管理，按上表逐文件手动回退
```
