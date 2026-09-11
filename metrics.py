from __future__ import annotations

import torch
from torch.nn import functional as F


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    data_range: float = 1.0,
) -> torch.Tensor:
    """按 scikit-image 的默认 RGB 设置计算每张图像的 SSIM。"""
    prediction = prediction.float()
    target = target.float()
    channels = prediction.shape[1]
    window = torch.full(
        (channels, 1, window_size, window_size),
        1.0 / (window_size * window_size),
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mu_prediction = F.conv2d(prediction, window, groups=channels)
    mu_target = F.conv2d(target, window, groups=channels)
    covariance_scale = (window_size * window_size) / (
        window_size * window_size - 1
    )
    variance_prediction = covariance_scale * (
        F.conv2d(prediction.square(), window, groups=channels)
        - mu_prediction.square()
    )
    variance_target = covariance_scale * (
        F.conv2d(target.square(), window, groups=channels)
        - mu_target.square()
    )
    covariance = covariance_scale * (
        F.conv2d(prediction * target, window, groups=channels)
        - mu_prediction * mu_target
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_prediction * mu_target + c1) * (2 * covariance + c2)
    denominator = (
        mu_prediction.square() + mu_target.square() + c1
    ) * (variance_prediction + variance_target + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean(dim=(1, 2, 3))


@torch.no_grad()
def batch_psnr_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算一个批次中每张图像的 PSNR 和 SSIM。"""
    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    ssim = ssim_per_image(prediction, target)
    return psnr, ssim
