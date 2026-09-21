# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

EPS = 1e-6
Center = Optional[Tuple[float, float]]


@dataclass(frozen=True)
class DDRDiscCleanConfig:
    """Parameters for DDR optic-disc highlight/contour suppression."""

    # 修复视盘中心的强高亮区域
    disc_radius: float = 70.0
    disc_bg_sigma: float = 35.0
    disc_fill_sigma: float = 55.0
    disc_feather_sigma: float = 14.0

    # 扩大处理范围
    highlight_radius_scale: float = 2.25
    highlight_fill_sigma: float = 110.0
    highlight_feather_sigma: float = 28.0
    highlight_strength: float = 1.0

    # 弱化视杯、视盘附近的高亮轮廓
    contour_radius_scale: float = 1.9
    contour_fill_sigma: float = 90.0
    contour_feather_sigma: float = 24.0
    contour_strength: float = 1.0


# 检查视盘高亮清理配置中的半径、平滑尺度和融合强度是否合法
def validate_config(config: DDRDiscCleanConfig) -> None:
    if config.disc_radius <= 0:
        raise ValueError("disc_radius must be positive")
    if config.disc_bg_sigma <= 0 or config.disc_fill_sigma <= 0:
        raise ValueError("disc_bg_sigma and disc_fill_sigma must be positive")
    if config.disc_feather_sigma <= 0:
        raise ValueError("disc_feather_sigma must be positive")
    if config.highlight_radius_scale <= 1.0:
        raise ValueError("highlight_radius_scale must be greater than 1")
    if config.highlight_fill_sigma <= 0 or config.highlight_feather_sigma <= 0:
        raise ValueError("highlight fill/feather sigma values must be positive")
    if not 0.0 <= config.highlight_strength <= 1.0:
        raise ValueError("highlight_strength must be between 0 and 1")
    if config.contour_radius_scale <= 1.0:
        raise ValueError("contour_radius_scale must be greater than 1")
    if config.contour_fill_sigma <= 0 or config.contour_feather_sigma <= 0:
        raise ValueError("contour fill/feather sigma values must be positive")
    if not 0.0 <= config.contour_strength <= 1.0:
        raise ValueError("contour_strength must be between 0 and 1")


# # 根据高斯 sigma 计算覆盖正负 3σ 的奇数卷积核尺寸
def odd_kernel_from_sigma(sigma: float) -> int:
    # 高斯核覆盖约正负 3σ 并通过按位或 1 保证核尺寸为奇数
    kernel_size = int(round(sigma * 6.0)) | 1
    return max(3, kernel_size)


# 对输入张量逐通道执行带安全边界填充的高斯模糊
def gaussian_blur_reflect(
        image: torch.Tensor,
        kernel_size: int,
        sigma: float,
) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    channels = image.shape[1]
    height, width = image.shape[-2:]
    radius = kernel_size // 2
    coords = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # 将二维高斯卷积分解为横向、纵向两个一维卷积 减少计算量
    kernel_x = kernel.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    kernel_y = kernel.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)

    # 图像尺寸不足以做反射填充时退化为边缘复制
    pad_mode = "reflect" if radius < min(height, width) else "replicate"
    padded = F.pad(image, (radius, radius, 0, 0), mode=pad_mode)
    blurred = F.conv2d(padded, kernel_x, groups=channels)
    padded = F.pad(blurred, (0, 0, radius, radius), mode=pad_mode)
    return F.conv2d(padded, kernel_y, groups=channels)


# 根据图像亮度生成眼底有效视野掩码并去除不稳定边缘
def retinal_mask(image: torch.Tensor) -> torch.Tensor:

    # 根据眼底区域与黑色背景的亮度差异 生成有效视野 FOV 掩膜
    gray = image.mean(dim=1, keepdim=True)
    mask = (gray > 0.03).float()

    # 轻微腐蚀掩膜边缘
    mask = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=5, stride=1, padding=2)
    return mask.clamp(0.0, 1.0)


# 使用掩膜外的有效像素进行归一化高斯插值以填补目标区域
def normalized_gaussian_fill(
        value: torch.Tensor,
        valid_mask: torch.Tensor,
        sigma: float,
) -> torch.Tensor:
    kernel_size = odd_kernel_from_sigma(sigma)
    valid = valid_mask.float()
    if valid.shape[1] == 1 and value.shape[1] != 1:
        valid = valid.expand_as(value)

    # 归一化卷积 只用掩膜外的有效像素估计局部背景
    numerator = gaussian_blur_reflect(value * valid, kernel_size, sigma)
    denominator = gaussian_blur_reflect(valid, kernel_size, sigma).clamp_min(EPS)
    return numerator / denominator


# 自动或按指定中心生成视盘高亮区域的羽化修复掩膜
def make_disc_repair_mask(
        ddr_tensor: torch.Tensor,
        illumination_b: torch.Tensor,
        disc_radius: float,
        bg_sigma: float,
        feather_sigma: float,
        center: Center = None,
) -> torch.Tensor:
    height, width = illumination_b.shape[-2:]
    fov = retinal_mask(ddr_tensor)

    if center is None:
        gray = illumination_b.mean(dim=1, keepdim=True)
        background = gaussian_blur_reflect(
            gray,
            odd_kernel_from_sigma(bg_sigma),
            bg_sigma,
        )
        residual = gray - background
        score = residual.masked_fill(fov <= 0.0, -1e6)
        peak_index = score.reshape(-1).argmax()
        peak_y = (peak_index // width).float()
        peak_x = (peak_index % width).float()
    else:
        peak_x = torch.tensor(float(center[0]), device=illumination_b.device)
        peak_y = torch.tensor(float(center[1]), device=illumination_b.device)

    # 以定位点为圆心生成视盘硬掩膜 并限制处理范围必须位于眼底有效视野内
    yy = torch.arange(height, device=illumination_b.device).view(1, 1, height, 1)
    xx = torch.arange(width, device=illumination_b.device).view(1, 1, 1, width)
    distance2 = (xx - peak_x) ** 2 + (yy - peak_y) ** 2
    hard_mask = (distance2 <= disc_radius ** 2).float() * fov

    # 高斯羽化掩膜边缘 避免修复区域与原 illumination 之间出现明显接缝
    soft_mask = gaussian_blur_reflect(
        hard_mask,
        odd_kernel_from_sigma(feather_sigma),
        feather_sigma,
    )
    max_value = soft_mask.amax().clamp_min(EPS)
    return (soft_mask / max_value).clamp(0.0, 1.0)


# 用周围正常 illumination 平滑替换 DDR 视盘中心的强高亮区域
def remove_ddr_disc_highlight(
        ddr_tensor: torch.Tensor,
        illumination_b: torch.Tensor,
        config: DDRDiscCleanConfig,
        center: Center = None,
) -> torch.Tensor:
    # 构造视盘核心高亮区域的软掩膜
    soft_mask = make_disc_repair_mask(
        ddr_tensor=ddr_tensor,
        illumination_b=illumination_b,
        disc_radius=config.disc_radius,
        bg_sigma=config.disc_bg_sigma,
        feather_sigma=config.disc_feather_sigma,
        center=center,
    )
    fov = retinal_mask(ddr_tensor)

    # 使用周围正常眼底像素估计应有的 illumination
    valid_for_fill = (fov * (1.0 - (soft_mask > 0.05).float())).clamp(0.0, 1.0)
    filled = normalized_gaussian_fill(
        illumination_b,
        valid_for_fill,
        config.disc_fill_sigma,
    )

    # 按软掩膜平滑融合原 illumination 与填补结果 完成中心高亮去除
    cleaned = illumination_b * (1.0 - soft_mask) + filled * soft_mask
    return cleaned.clamp(0.0, 1.0)


# 压制 DDR 视盘中心外侧仍高于局部背景的宽范围亮度残留
def suppress_ddr_broad_highlight(
        ddr_tensor: torch.Tensor,
        illumination_b: torch.Tensor,
        illumination_clean: torch.Tensor,
        config: DDRDiscCleanConfig,
        center: Center = None,
) -> torch.Tensor:
    highlight_mask = make_disc_repair_mask(
        ddr_tensor=ddr_tensor,
        illumination_b=illumination_b,
        disc_radius=config.disc_radius * config.highlight_radius_scale,
        bg_sigma=config.disc_bg_sigma,
        feather_sigma=config.highlight_feather_sigma,
        center=center,
    )
    fov = retinal_mask(ddr_tensor)
    valid_for_fill = (fov * (1.0 - (highlight_mask > 0.02).float())).clamp(0.0, 1.0)
    local_background = normalized_gaussian_fill(
        illumination_clean,
        valid_for_fill,
        config.highlight_fill_sigma,
    )
    blend = (highlight_mask * config.highlight_strength).clamp(0.0, 1.0)
    bright_excess = (illumination_clean - local_background).clamp_min(0.0)
    corrected = illumination_clean - bright_excess * blend
    return corrected.clamp(0.0, 1.0)


# 平滑 DDR 视盘和视杯附近残留的环状亮度轮廓
def suppress_ddr_disc_contour(
        ddr_tensor: torch.Tensor,
        illumination_b: torch.Tensor,
        illumination_clean: torch.Tensor,
        config: DDRDiscCleanConfig,
        center: Center = None,
) -> torch.Tensor:
    
    # 在视盘/视杯轮廓附近构造更宽的软掩膜 处理残留的环状边界
    contour_mask = make_disc_repair_mask(
        ddr_tensor=ddr_tensor,
        illumination_b=illumination_b,
        disc_radius=config.disc_radius * config.contour_radius_scale,
        bg_sigma=config.disc_bg_sigma,
        feather_sigma=config.contour_feather_sigma,
        center=center,
    )
    fov = retinal_mask(ddr_tensor)
    valid_for_fill = (fov * (1.0 - (contour_mask > 0.03).float())).clamp(0.0, 1.0)
    filled = normalized_gaussian_fill(
        illumination_clean,
        valid_for_fill,
        config.contour_fill_sigma,
    )
    blend = (contour_mask * config.contour_strength).clamp(0.0, 1.0)

    # 用外围背景的平滑估计替换轮廓区域 同时保留掩膜外原有的全局光照
    flattened = illumination_clean * (1.0 - blend) + filled * blend
    return flattened.clamp(0.0, 1.0)


def _validate_inputs(ddr_tensor: torch.Tensor, illumination_b: torch.Tensor) -> None:
    if ddr_tensor.ndim != 4 or illumination_b.ndim != 4:
        raise ValueError("ddr_tensor and illumination_b must be [N, 3, H, W] tensors")
    if ddr_tensor.shape != illumination_b.shape:
        raise ValueError(
            "ddr_tensor and illumination_b must have the same shape; "
            f"got {tuple(ddr_tensor.shape)} and {tuple(illumination_b.shape)}"
        )
    if ddr_tensor.shape[1] != 3:
        raise ValueError(f"Expected RGB tensors with 3 channels, got {ddr_tensor.shape}")


# 依次清理视盘中心高亮、外围宽高亮和视杯视盘轮廓并返回最终 illumination
def clean_ddr_illumination(
        ddr_tensor: torch.Tensor,
        illumination_b: torch.Tensor,
        config: Optional[DDRDiscCleanConfig] = None,
        disc_center: Center = None,
        return_debug: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    config = config or DDRDiscCleanConfig()
    validate_config(config)
    _validate_inputs(ddr_tensor, illumination_b)

    # 第一步 去除视盘中心最强的局部高亮
    disc_clean = remove_ddr_disc_highlight(
        ddr_tensor=ddr_tensor,
        illumination_b=illumination_b,
        config=config,
        center=disc_center,
    )

    # 第二步 压制中心外侧仍然存在的宽范围亮度残留
    highlight_clean = suppress_ddr_broad_highlight(
        ddr_tensor=ddr_tensor,
        illumination_b=illumination_b,
        illumination_clean=disc_clean,
        config=config,
        center=disc_center,
    )

    # 第三步 进一步抹平视盘/视杯边界轮廓 得到供 Retinex swap 使用的结果
    contour_clean = suppress_ddr_disc_contour(
        ddr_tensor=ddr_tensor,
        illumination_b=illumination_b,
        illumination_clean=highlight_clean,
        config=config,
        center=disc_center,
    )

    if not return_debug:
        return contour_clean
    return contour_clean, {
        "disc_clean": disc_clean,
        "highlight_clean": highlight_clean,
    }


class DDRDiscCleaner(torch.nn.Module):

    def __init__(self, config: Optional[DDRDiscCleanConfig] = None) -> None:
        super().__init__()
        self.config = config or DDRDiscCleanConfig()
        validate_config(self.config)

    def forward(
            self,
            ddr_tensor: torch.Tensor,
            illumination_b: torch.Tensor,
            disc_center: Center = None,
    ) -> torch.Tensor:
        return clean_ddr_illumination(
            ddr_tensor=ddr_tensor,
            illumination_b=illumination_b,
            config=self.config,
            disc_center=disc_center,
        )
