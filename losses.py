from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _gaussian_window(
    window_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, dtype=torch.float32, device=device)
    coordinates -= window_size // 2
    gaussian = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    gaussian /= gaussian.sum()
    return gaussian[:, None] * gaussian[None, :]


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """计算批次内每张图像的可微结构相似度。"""
    prediction = prediction.float()
    target = target.float()
    channels = prediction.shape[1]
    window = _gaussian_window(window_size, 1.5, prediction.device)
    window = window.expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    mu_prediction = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_prediction * mu_target

    sigma_prediction = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mu_prediction_sq
    )
    sigma_target = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mu_target_sq
    )
    sigma_cross = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mu_cross
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        sigma_prediction + sigma_target + c2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean(dim=(1, 2, 3))


class CharbonnierLoss(nn.Module):
    """使用平滑形式计算像素重建误差。"""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.epsilon**2).mean()


class EnhancementLoss(nn.Module):
    """组合 Charbonnier 损失和结构相似度损失。"""

    def __init__(self, ssim_weight: float = 0.2, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss(epsilon)
        self.ssim_weight = ssim_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float()
        target = target.float()
        charbonnier = self.charbonnier(prediction, target)
        ssim_loss = 1.0 - ssim_per_image(
            prediction.clamp(0.0, 1.0),
            target.clamp(0.0, 1.0),
        ).mean()
        return charbonnier + self.ssim_weight * ssim_loss
