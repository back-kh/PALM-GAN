"""Loss functions used by PALM-GAN training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def structural_similarity_loss(
    prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """Differentiable single-scale SSIM loss for normalized grayscale images."""
    padding = window_size // 2
    mean_prediction = F.avg_pool2d(prediction, window_size, 1, padding)
    mean_target = F.avg_pool2d(target, window_size, 1, padding)
    variance_prediction = (
        F.avg_pool2d(prediction.square(), window_size, 1, padding)
        - mean_prediction.square()
    )
    variance_target = (
        F.avg_pool2d(target.square(), window_size, 1, padding) - mean_target.square()
    )
    covariance = (
        F.avg_pool2d(prediction * target, window_size, 1, padding)
        - mean_prediction * mean_target
    )
    constant1, constant2 = 0.01**2, 0.03**2
    similarity = (
        (2.0 * mean_prediction * mean_target + constant1)
        * (2.0 * covariance + constant2)
        / (
            (mean_prediction.square() + mean_target.square() + constant1)
            * (variance_prediction + variance_target + constant2)
        ).clamp_min(1e-8)
    )
    return 1.0 - similarity.mean()


def edge_consistency_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Match differentiable morphological edges to preserve faint strokes."""

    def boundary(values: torch.Tensor) -> torch.Tensor:
        dilation = F.max_pool2d(values, 3, stride=1, padding=1)
        erosion = -F.max_pool2d(-values, 3, stride=1, padding=1)
        return dilation - erosion

    return F.l1_loss(boundary(prediction), boundary(target))
