import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TextDegradationPipeline:
    """文本图像专项退化管线。

    模拟真实场景中文本图像的多种退化类型：
    - 运动模糊（相机抖动/文字移动）
    - JPEG 压缩伪影
    - 局部遮挡（手指/阴影）
    - 扫描噪声（纸张纹理、摩尔纹、椒盐噪声）
    - 散焦模糊（失焦）

    使用方式：pipeline = TextDegradationPipeline(prob_motion=0.3, prob_jpeg=0.5, ...)
              degraded_img = pipeline.apply(img)
    """

    def __init__(
        self,
        prob_motion=0.0,
        prob_jpeg=0.0,
        prob_occlusion=0.0,
        prob_scan_noise=0.0,
        prob_defocus=0.0,
        jpeg_quality_range=(30, 70),
        occlusion_ratio=(0.02, 0.12),
        motion_kernel_range=(3, 9),
        defocus_sigma_range=(1.5, 4.0),
        scan_noise_density=(0.001, 0.008),
        seed=None,
    ):
        self.rng = np.random.RandomState(seed)
        self.prob_motion = float(prob_motion)
        self.prob_jpeg = float(prob_jpeg)
        self.prob_occlusion = float(prob_occlusion)
        self.prob_scan_noise = float(prob_scan_noise)
        self.prob_defocus = float(prob_defocus)
        self.jpeg_quality_range = tuple(jpeg_quality_range)
        self.occlusion_ratio = tuple(occlusion_ratio)
        self.motion_kernel_range = tuple(motion_kernel_range)
        self.defocus_sigma_range = tuple(defocus_sigma_range)
        self.scan_noise_density = tuple(scan_noise_density)

    def _motion_blur(self, img):
        ksize = self.rng.randint(self.motion_kernel_range[0], self.motion_kernel_range[1] + 1)
        if ksize % 2 == 0:
            ksize += 1
        angle = self.rng.uniform(0, 180)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        cx, cy = ksize // 2, ksize // 2
        rad = np.deg2rad(angle)
        for i in range(ksize):
            offset = int(round((i - cy) * np.tan(rad)))
            if 0 <= cx + offset < ksize:
                kernel[cx + offset, i] = 1.0
        kernel = kernel / kernel.sum()
        return cv2.filter2D(img, -1, kernel)

    def _jpeg_compression(self, img):
        quality = self.rng.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1] + 1)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _partial_occlusion(self, img):
        h, w = img.shape[:2]
        area_min = h * w * self.occlusion_ratio[0]
        area_max = h * w * self.occlusion_ratio[1]
        n_rects = self.rng.randint(1, 4)
        result = img.copy().astype(np.float32)
        for _ in range(n_rects):
            rw = self.rng.randint(int(w * 0.05), int(w * 0.35))
            rh = self.rng.randint(int(h * 0.03), int(h * 0.20))
            rx = self.rng.randint(0, max(w - rw, 1))
            ry = self.rng.randint(0, max(h - rh, 1))
            brightness = self.rng.uniform(30, 120)
            result[ry:ry + rh, rx:rx + rw] = brightness
        return np.clip(result, 0, 255).astype(np.uint8)

    def _scan_noise(self, img):
        result = img.astype(np.float32)
        density = self.rng.uniform(*self.scan_noise_density)
        sp_noise = self.rng.choice([0, 255], size=img.shape, p=[1 - density, density]).astype(np.float32)
        mask = self.rng.random(img.shape) < density
        result[mask] = sp_noise[mask]
        moire_strength = self.rng.uniform(2, 8)
        freq_x = self.rng.uniform(0.05, 0.15)
        freq_y = self.rng.uniform(0.05, 0.15)
        y_coords, x_coords = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        moire = moire_strength * np.sin(2 * np.pi * (freq_x * x_coords + freq_y * y_coords)).astype(np.float32)
        result = result +moire
        return np.clip(result, 0, 255).astype(np.uint8)

    def _defocus_blur(self, img):
        sigma = self.rng.uniform(*self.defocus_sigma_range)
        ksize = int(sigma * 3) | 1
        ksize = max(ksize, 3)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    def apply(self, img):
        if self.rng.random() < self.prob_motion:
            img = self._motion_blur(img)
        if self.rng.random() < self.prob_jpeg:
            img = self._jpeg_compression(img)
        if self.rng.random() < self.prob_occlusion:
            img = self._partial_occlusion(img)
        if self.rng.random() < self.prob_scan_noise:
            img = self._scan_noise(img)
        if self.rng.random() < self.prob_defocus:
            img = self._defocus_blur(img)
        return img

    @staticmethod
    def get_preset(name):
        presets = {
            "none": {},
            "light": dict(
                prob_motion=0.10, prob_jpeg=0.25, prob_occlusion=0.05,
                prob_scan_noise=0.05, prob_defocus=0.10,
                jpeg_quality_range=(50, 85), occlusion_ratio=(0.01, 0.06),
            ),
            "medium": dict(
                prob_motion=0.25, prob_jpeg=0.45, prob_occlusion=0.12,
                prob_scan_noise=0.15, prob_defocus=0.22,
                jpeg_quality_range=(30, 75), occlusion_ratio=(0.02, 0.12),
            ),
            "heavy": dict(
                prob_motion=0.40, prob_jpeg=0.65, prob_occlusion=0.22,
                prob_scan_noise=0.30, prob_defocus=0.35,
                jpeg_quality_range=(15, 55), occlusion_ratio=(0.04, 0.18),
            ),
            "text_realistic": dict(
                prob_motion=0.20, prob_jpeg=0.50, prob_occlusion=0.08,
                prob_scan_noise=0.20, prob_defocus=0.18,
                jpeg_quality_range=(35, 70), occlusion_ratio=(0.02, 0.10),
                motion_kernel_range=(3, 7), defocus_sigma_range=(1.5, 3.5),
            ),
        }
        cfg = presets.get(name, presets["text_realistic"])
        return TextDegradationPipeline(**cfg)


class TextImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        # 只读取常见的图片格式
        self.file_names = [f for f in os.listdir(hr_dir) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        hr_path = os.path.join(self.hr_dir, file_name)
        lr_path = os.path.join(self.lr_dir, file_name)

        # 1. 读取图片
        img_hr = cv2.imread(hr_path)
        img_lr = cv2.imread(lr_path)

        # 2. 容错
        if img_hr is None or img_lr is None:
            # 读不到就给黑图，防止崩坏
            img_hr = np.zeros((512, 512, 3), dtype=np.uint8)
            img_lr = np.zeros((128, 128, 3), dtype=np.uint8)

        # 3. 转 RGB
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)

        # 4. 强制统一尺寸 (LR=128, HR=512)
        # 这一步是为了防止“Stack Error”
        img_lr = cv2.resize(img_lr, (128, 128), interpolation=cv2.INTER_CUBIC)
        img_hr = cv2.resize(img_hr, (512, 512), interpolation=cv2.INTER_CUBIC)

        # 5. 【关键修复】手动强制转 Tensor 并掉头 (H, W, C) -> (C, H, W)
        # 这种写法最稳，绝对不会错
        # 原代码是 / 255.0
        # 新代码：(数值 / 127.5) - 1.0  ---> 这样范围就变成了 -1.0 到 1.0
        img_lr = (torch.from_numpy(img_lr).permute(2, 0, 1).float() / 127.5) - 1.0
        img_hr = (torch.from_numpy(img_hr).permute(2, 0, 1).float() / 127.5) - 1.0

        return {'LR': img_lr, 'HR': img_hr}


# ==========================================
# 新增：TextSRDataset
# 功能：输入原始 HR 图像，动态生成对应的 LR（下采样 + 可选模糊 + 噪声），返回 Tensor
# 适用于文本图像超分任务（例如 HR:256 -> LR:64）
# ==========================================
class TextSRDataset(Dataset):
    def __init__(
            self,
            hr_dir,
            lr_dir=None,
            mask_dir=None,
            scale=4,
            hr_size=256,
            augment=False,
            blur_prob=0.5,
            noise_std=0.0,
            degradation="none",
            **deg_kwargs):
        """
        hr_dir: 存放 HR 图像的目录
        scale: 下采样倍率 (例如 4 表示 256 -> 64)
        hr_size: 输出的 HR 尺寸 (会把原图 resize/crop 到 hr_size)
        augment: 是否打开数据增强（对文本有用的二值化/扰动）
        blur_prob: 生成 LR 时加入高斯模糊的概率
        noise_std: 在 LR 上加入高斯噪声的标准差 (0.0 表示不加)
        degradation: 文本退化管线预设名 ("none"/"light"/"medium"/"heavy"/"text_realistic")
                     或 TextDegradationPipeline 实例
        **deg_kwargs: 传给 TextDegradationPipeline 的额外参数
        """
        self.hr_dir = hr_dir
        self.file_names = [f for f in os.listdir(hr_dir) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        self.file_names.sort()
        self.lr_dir = lr_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.hr_size = hr_size if isinstance(hr_size, int) else hr_size[0]
        self.augment = augment
        self.blur_prob = blur_prob
        self.noise_std = noise_std

        if isinstance(degradation, TextDegradationPipeline):
            self.degradation_pipeline = degradation
        elif isinstance(degradation, str):
            self.degradation_pipeline = TextDegradationPipeline.get_preset(degradation)
        else:
            self.degradation_pipeline = TextDegradationPipeline()

    def __len__(self):
        return len(self.file_names)

    def _make_lr(self, img_hr):
        # 输入 img_hr 为 numpy RGB uint8
        # 1) 先保证 HR 是 hr_size
        img_hr = cv2.resize(img_hr, (self.hr_size, self.hr_size), interpolation=cv2.INTER_CUBIC)

        # 2) 下采样生成 LR
        lr_size = self.hr_size // self.scale
        img_lr = cv2.resize(img_hr, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)

        # 3) 随机高斯模糊（传统退化）
        if np.random.rand() < self.blur_prob:
            k = 3 if np.random.rand() < 0.7 else 5
            sigma = np.random.uniform(0.2, 1.5)
            img_lr = cv2.GaussianBlur(img_lr, (k, k), sigmaX=sigma)

        # 4) 可选噪声（传统退化）
        if self.noise_std > 0:
            noise = np.random.randn(*img_lr.shape) * (self.noise_std * 255.0)
            img_lr = img_lr.astype(np.float32) + noise
            img_lr = np.clip(img_lr, 0, 255).astype(np.uint8)

        # 5) 文本专项退化管线（WEEK_3 新增：运动模糊/压缩/遮挡/扫描噪声/散焦）
        img_lr = self.degradation_pipeline.apply(img_lr)

        return img_hr, img_lr

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        hr_path = os.path.join(self.hr_dir, file_name)

        img_hr = cv2.imread(hr_path)
        if img_hr is None:
            # 返回空白图（避免训练中断）
            img_hr = np.zeros((self.hr_size, self.hr_size, 3), dtype=np.uint8)

        # BGR -> RGB
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

        # 可选增强（针对文本图像：二值化/形态学变换/亮度对比度扰动）
        # 注意：当使用配对 LR 数据时，不能只增强 HR，否则会破坏 LR-HR 对齐监督。
        if self.augment and self.lr_dir is None:
            # 随机对比度/亮度
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-10, 10)
            img_hr = np.clip(img_hr * alpha + beta, 0, 255).astype(np.uint8)

            # 小概率做二值化（模拟扫描文本）
            if np.random.rand() < 0.2:
                gray = cv2.cvtColor(img_hr, cv2.COLOR_RGB2GRAY)
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_hr = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

        if self.lr_dir is not None:
            lr_path = os.path.join(self.lr_dir, file_name)
            img_lr = cv2.imread(lr_path)
            if img_lr is None:
                # 兼容后缀不一致，尝试同 stem 的其他后缀
                stem = os.path.splitext(file_name)[0]
                for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                    alt = os.path.join(self.lr_dir, stem + ext)
                    img_lr = cv2.imread(alt)
                    if img_lr is not None:
                        break

            if img_lr is None:
                # LR 缺失时回退到在线退化，保证训练不中断
                img_hr, img_lr = self._make_lr(img_hr)
            else:
                # 关键：cv2 读取为 BGR，这里统一转为 RGB，保持与 HR/在线退化路径一致。
                img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
                # 使用预生成 LR，同时统一尺寸
                img_hr = cv2.resize(img_hr, (self.hr_size, self.hr_size), interpolation=cv2.INTER_CUBIC)
                lr_size = self.hr_size // self.scale
                img_lr = cv2.resize(img_lr, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        else:
            img_hr, img_lr = self._make_lr(img_hr)

        # 转为 Tensor 并归一化到 [-1, 1]
        img_lr_t = (torch.from_numpy(img_lr).permute(2, 0, 1).float() / 127.5) - 1.0
        img_hr_t = (torch.from_numpy(img_hr).permute(2, 0, 1).float() / 127.5) - 1.0

        # 新增：尝试加载对应的 mask（dataset/masks/<filename>），如果存在则返回 'mask'
        if self.mask_dir is not None:
            mask_path_options = [
                os.path.join(self.mask_dir, file_name),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.png'),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.jpg'),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.jpeg'),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.bmp'),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.webp'),
            ]
        else:
            mask_path_options = [
                os.path.join(os.path.dirname(self.hr_dir), 'masks', file_name),  # sibling /dataset/masks
                os.path.join(self.hr_dir, '..', 'masks', file_name),  # another attempt
                os.path.join('dataset', 'masks', file_name)  # fallback
            ]
        mask_tensor = None
        for mp in mask_path_options:
            mp_abs = os.path.abspath(mp)
            if os.path.exists(mp_abs):
                try:
                    m = cv2.imread(mp_abs, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        # ensure same size as HR; resize if needed
                        if (m.shape[0] != self.hr_size) or (m.shape[1] != self.hr_size):
                            m = cv2.resize(m, (self.hr_size, self.hr_size), interpolation=cv2.INTER_NEAREST)
                        # convert to tensor [1, H, W]
                        m_t = torch.from_numpy(m).unsqueeze(0)
                        mask_tensor = m_t
                        break
                except Exception:
                    # ignore read errors and continue
                    pass

        ret = {'LR': img_lr_t, 'HR': img_hr_t, 'fname': file_name}
        if mask_tensor is not None:
            ret['mask'] = mask_tensor

        return ret
