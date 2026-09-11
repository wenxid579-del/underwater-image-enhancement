from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """在通道维度上执行二维层归一化。"""

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=1, keepdim=True)
        variance = (inputs - mean).square().mean(dim=1, keepdim=True)
        normalized = (inputs - mean) * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


class CascadedValueRefinementAttention(nn.Module):
    """通过级联局部细化增强值分支的通道转置注意力。"""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("通道数必须能够被注意力头数整除")
        self.heads = heads
        self.log_temperature = nn.Parameter(torch.zeros(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.qkv_local = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
        )
        self.value_local = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
        )
        self.value_context = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
        )
        selector_channels = max(channels // 4, 4)
        self.scale_selector = nn.Sequential(
            nn.Linear(channels, selector_channels),
            nn.GELU(),
            nn.Linear(selector_channels, heads * 2),
        )
        self.refinement_scale = nn.Parameter(torch.zeros(1))
        self.local_bypass_scale = nn.Parameter(torch.zeros(1))
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        head_channels = channels // self.heads
        query, key, value = self.qkv_local(self.qkv(features)).chunk(3, dim=1)

        def as_attention_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.heads, head_channels, height * width)

        def as_feature_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.heads, head_channels, height, width)

        value_local = self.value_local(value)
        value_context = self.value_context(F.gelu(value + value_local))
        scale_weights = self.scale_selector(features.mean(dim=(2, 3))).reshape(
            batch, self.heads, 2, 1, 1
        )
        scale_weights = scale_weights.softmax(dim=2)
        local_heads = (
            scale_weights[:, :, 0:1] * as_feature_heads(value_local)
            + scale_weights[:, :, 1:2] * as_feature_heads(value_context)
        )
        refined_local = local_heads.reshape(batch, channels, height, width)
        refined_value = value + torch.tanh(self.refinement_scale) * refined_local

        query = F.normalize(as_attention_heads(query), dim=-1)
        key = F.normalize(as_attention_heads(key), dim=-1)
        attention = (
            query @ key.transpose(-2, -1)
        ) * self.log_temperature.exp().unsqueeze(0)
        attention = attention.softmax(dim=-1)
        global_output = (attention @ as_attention_heads(refined_value)).reshape(
            batch, channels, height, width
        )
        output = global_output + torch.tanh(self.local_bypass_scale) * refined_local
        return self.project(output)


class FrequencyDecoupledModulationFFN(nn.Module):
    """分离低频与高频成分，并利用交叉门控完成特征调制。"""

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        modulation_channels = max(channels // 4, 4)
        self.project_in = nn.Conv2d(channels, hidden_channels, 1)
        self.low_refine = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            5,
            padding=2,
            groups=hidden_channels,
        )
        self.high_refine = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
        )
        self.low_modulator = nn.Sequential(
            nn.Conv2d(hidden_channels, modulation_channels, 1),
            nn.GELU(),
            nn.Conv2d(modulation_channels, hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.high_modulator = nn.Sequential(
            nn.Conv2d(hidden_channels, modulation_channels, 1),
            nn.GELU(),
            nn.Conv2d(modulation_channels, hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.low_scale = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.high_scale = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.project_out = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expanded = self.project_in(features)
        low = F.avg_pool2d(expanded, kernel_size=5, stride=1, padding=2)
        high = expanded - low
        refined_low = self.low_refine(low)
        refined_high = self.high_refine(high)
        low_gate = self.low_modulator(high.mean(dim=(2, 3), keepdim=True))
        high_gate = self.high_modulator(low.mean(dim=(2, 3), keepdim=True))
        transformed = (
            expanded
            + torch.tanh(self.low_scale) * low_gate * refined_low
            + torch.tanh(self.high_scale) * high_gate * refined_high
        )
        return self.project_out(F.gelu(transformed))


class AquaFormerBlock(nn.Module):
    """由 CVRA 和 FDM-FFN 组成的基本特征变换单元。"""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = CascadedValueRefinementAttention(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = FrequencyDecoupledModulationFFN(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features + self.attention(self.norm1(features))
        return features + self.ffn(self.norm2(features))


class Downsample(nn.Sequential):
    """使用深度卷积降低空间分辨率，再调整通道数。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=2,
                padding=1,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, 1),
        )


class Upsample(nn.Module):
    """将特征恢复到指定尺寸并调整通道数。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self, features: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        features = F.interpolate(
            features,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.project(features)


class DetailCrossAttentionFusion(nn.Module):
    """在普通跳跃连接上叠加细节交叉注意力校正。"""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("通道数必须能够被注意力头数整除")
        self.heads = heads
        self.log_temperature = nn.Parameter(torch.zeros(heads, 1, 1))
        self.fuse = nn.Conv2d(channels * 2, channels, 1)
        self.query = nn.Conv2d(channels, channels, 1)
        self.query_local = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
        )
        self.key_value = nn.Conv2d(channels, channels * 2, 1)
        self.key_value_local = nn.Conv2d(
            channels * 2,
            channels * 2,
            3,
            padding=1,
            groups=channels * 2,
        )
        self.project = nn.Conv2d(channels, channels, 1)
        self.correction_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        decoder_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        base = self.fuse(torch.cat((decoder_features, encoder_features), dim=1))
        encoder_low = F.avg_pool2d(
            encoder_features,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        encoder_detail = encoder_features - encoder_low
        query = self.query_local(self.query(decoder_features))
        key, value = self.key_value_local(
            self.key_value(encoder_detail)
        ).chunk(2, dim=1)

        batch, channels, height, width = query.shape
        head_channels = channels // self.heads

        def as_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.heads, head_channels, height * width)

        query = F.normalize(as_heads(query), dim=-1)
        key = F.normalize(as_heads(key), dim=-1)
        value = as_heads(value)
        attention = (
            query @ key.transpose(-2, -1)
        ) * self.log_temperature.exp().unsqueeze(0)
        attention = attention.softmax(dim=-1)
        correction = (attention @ value).reshape(batch, channels, height, width)
        correction = self.project(correction)
        return base + torch.tanh(self.correction_scale) * correction


class AquaLiteFormer(nn.Module):
    """用于水下图像增强的轻量级编码器-解码器网络。"""

    def __init__(
        self,
        base_channels: int = 16,
        depths: Sequence[int] = (1, 2, 4, 2, 1),
    ) -> None:
        super().__init__()
        if len(depths) != 5:
            raise ValueError("depths 必须依次给出五个阶段的模块数量")
        depths = tuple(int(depth) for depth in depths)
        if base_channels <= 0 or any(depth <= 0 for depth in depths):
            raise ValueError("基础通道数和各阶段深度必须为正整数")

        widths = (base_channels, base_channels * 2, base_channels * 4)
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1)

        def make_stage(channels: int, heads: int, depth: int) -> nn.ModuleList:
            return nn.ModuleList(
                AquaFormerBlock(channels, heads) for _ in range(depth)
            )

        self.encoder1 = make_stage(widths[0], 1, depths[0])
        self.down1 = Downsample(widths[0], widths[1])
        self.encoder2 = make_stage(widths[1], 2, depths[1])
        self.down2 = Downsample(widths[1], widths[2])
        self.bottleneck = make_stage(widths[2], 4, depths[2])
        self.up2 = Upsample(widths[2], widths[1])
        self.skip2 = DetailCrossAttentionFusion(widths[1], heads=2)
        self.decoder2 = make_stage(widths[1], 2, depths[3])
        self.up1 = Upsample(widths[1], widths[0])
        self.skip1 = DetailCrossAttentionFusion(widths[0], heads=1)
        self.decoder1 = make_stage(widths[0], 1, depths[4])
        self.tail = nn.Conv2d(widths[0], 3, 3, padding=1)
        nn.init.normal_(self.tail.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.tail.bias)

    @staticmethod
    def run_stage(
        features: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        for block in blocks:
            features = block(features)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoder1 = self.run_stage(self.stem(images), self.encoder1)
        encoder2 = self.run_stage(self.down1(encoder1), self.encoder2)
        features = self.run_stage(self.down2(encoder2), self.bottleneck)
        features = self.skip2(self.up2(features, encoder2.shape[-2:]), encoder2)
        features = self.run_stage(features, self.decoder2)
        features = self.skip1(self.up1(features, encoder1.shape[-2:]), encoder1)
        features = self.run_stage(features, self.decoder1)
        return images + self.tail(features)
