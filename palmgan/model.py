from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormActivation(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class ResidualConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )
        self.block = nn.Sequential(
            ConvNormActivation(input_channels, output_channels),
            ConvNormActivation(output_channels, output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.shortcut(inputs) + self.block(inputs)


class WindowTransformer2d(nn.Module):
    """Pre-norm spatial attention with bounded high-resolution memory."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        window_size: int | None,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be positive or None")
        self.window_size = window_size
        self.positional_encoding = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * mlp_ratio, channels),
            nn.Dropout(dropout),
        )

    def _partition(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
        batch, channels, height, width = inputs.shape
        window = self.window_size or max(height, width)
        pad_height = (window - height % window) % window
        pad_width = (window - width % window) % window
        padded = F.pad(inputs, (0, pad_width, 0, pad_height))
        padded_height, padded_width = padded.shape[-2:]
        windows = (
            padded.reshape(
                batch,
                channels,
                padded_height // window,
                window,
                padded_width // window,
                window,
            )
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, window * window, channels)
        )
        return windows, (batch, height, width, padded_height, padded_width)

    def _restore(
        self, windows: torch.Tensor, geometry: tuple[int, int, int, int, int]
    ) -> torch.Tensor:
        batch, height, width, padded_height, padded_width = geometry
        channels = windows.shape[-1]
        window = int(windows.shape[1] ** 0.5)
        restored = (
            windows.reshape(
                batch,
                padded_height // window,
                padded_width // window,
                window,
                window,
                channels,
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, padded_height, padded_width)
        )
        return restored[:, :, :height, :width]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        positioned = inputs + self.positional_encoding(inputs)
        windows, geometry = self._partition(positioned)
        normalized = self.norm1(windows)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        windows = windows + attended
        windows = windows + self.mlp(self.norm2(windows))
        return self._restore(windows, geometry)


class AttentionResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        heads: int,
        window_size: int | None,
    ) -> None:
        super().__init__()
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )
        self.pre_attention = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.transformer = WindowTransformer2d(
            output_channels,
            heads=heads,
            window_size=window_size,
            mlp_ratio=4,
            dropout=0.1,
        )
        self.output = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.pre_attention(inputs)
        values = self.transformer(values)
        return self.shortcut(inputs) + self.output(values)


class PalmGANGenerator(nn.Module):
    """Multi-scale residual CNN generator with Transformer context blocks.

    The model returns logits. Apply ``torch.sigmoid`` to obtain a normalized
    clean-image or binarization probability map.
    """

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        channels = [base_channels * (2**index) for index in range(5)]
        heads = [max(1, channel // 64) for channel in channels]
        self.pool = nn.MaxPool2d(2)

        self.encoder1 = nn.Sequential(
            ConvNormActivation(1, channels[0]),
            ResidualConvBlock(channels[0], channels[0]),
        )
        self.encoder2 = nn.Sequential(
            ConvNormActivation(channels[0], channels[1]),
            ResidualConvBlock(channels[1], channels[1]),
        )
        self.encoder3 = nn.Sequential(
            ConvNormActivation(channels[1], channels[2]),
            ResidualConvBlock(channels[2], channels[2]),
        )
        self.encoder4_input = ConvNormActivation(channels[2], channels[3])
        self.encoder4_attention = AttentionResidualBlock(
            channels[3], channels[3], heads=heads[3], window_size=16
        )
        self.bottleneck_input = ConvNormActivation(channels[3], channels[4])
        self.bottleneck_attention = AttentionResidualBlock(
            channels[4], channels[4], heads=heads[4], window_size=None
        )

        self.up4 = nn.ConvTranspose2d(
            channels[4],
            channels[3],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.decoder4 = nn.Sequential(
            ConvNormActivation(channels[3] * 2, channels[3]),
            ResidualConvBlock(channels[3], channels[3]),
        )
        self.up3 = nn.ConvTranspose2d(
            channels[3],
            channels[2],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.decoder3_input = ConvNormActivation(channels[2] * 2, channels[2])
        self.decoder3_attention = AttentionResidualBlock(
            channels[2], channels[2], heads=heads[2], window_size=8
        )
        self.up2 = nn.ConvTranspose2d(
            channels[2],
            channels[1],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.decoder2 = nn.Sequential(
            ConvNormActivation(channels[1] * 2, channels[1]),
            ResidualConvBlock(channels[1], channels[1]),
        )
        self.up1 = nn.ConvTranspose2d(
            channels[1],
            channels[0],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.decoder1_input = ConvNormActivation(channels[0] * 2, channels[0])
        self.decoder1_attention = AttentionResidualBlock(
            channels[0], channels[0], heads=heads[0], window_size=8
        )
        self.output = nn.Conv2d(channels[0], 1, 1)

    @staticmethod
    def _join(values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if values.shape[-2:] != skip.shape[-2:]:
            values = F.interpolate(
                values, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([values, skip], dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("PALM-GAN expects input shaped [batch, 1, height, width]")
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4_attention(self.encoder4_input(self.pool(encoder3)))
        bottleneck = self.bottleneck_attention(
            self.bottleneck_input(self.pool(encoder4))
        )

        decoder4 = self.decoder4(self._join(self.up4(bottleneck), encoder4))
        decoder3 = self.decoder3_attention(
            self.decoder3_input(self._join(self.up3(decoder4), encoder3))
        )
        decoder2 = self.decoder2(self._join(self.up2(decoder3), encoder2))
        decoder1 = self.decoder1_attention(
            self.decoder1_input(self._join(self.up1(decoder2), encoder1))
        )
        return self.output(decoder1)


class PalmGANPatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator for degraded/clean image pairs."""

    def __init__(
        self, base_channels: int = 64, *, spectral_normalization: bool = True
    ) -> None:
        super().__init__()

        def convolution(*args: int, **kwargs: int | bool) -> nn.Module:
            layer = nn.Conv2d(*args, **kwargs)
            return nn.utils.spectral_norm(layer) if spectral_normalization else layer

        def block(
            input_channels: int,
            output_channels: int,
            *,
            use_norm: bool,
            dropout: float,
        ) -> nn.Sequential:
            return nn.Sequential(
                convolution(
                    input_channels,
                    output_channels,
                    3,
                    stride=2,
                    padding=1,
                    bias=not use_norm,
                ),
                nn.BatchNorm2d(output_channels) if use_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout) if dropout else nn.Identity(),
            )

        self.features = nn.Sequential(
            block(2, base_channels, use_norm=False, dropout=0.25),
            block(base_channels, base_channels * 2, use_norm=True, dropout=0.0),
            block(base_channels * 2, base_channels * 4, use_norm=False, dropout=0.25),
            convolution(base_channels * 4, base_channels * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.output = convolution(base_channels * 8, 1, 3, padding=1)

    def forward(self, degraded: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if degraded.shape != clean.shape:
            raise ValueError("degraded and clean tensors must have identical shapes")
        return self.output(self.features(torch.cat([degraded, clean], dim=1)))
