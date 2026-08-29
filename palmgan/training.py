"""Framework-neutral PALM-GAN training step built on PyTorch tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .losses import dice_loss, edge_consistency_loss, structural_similarity_loss
from .model import PalmGANGenerator, PalmGANPatchDiscriminator


@dataclass(frozen=True)
class LossWeights:
    """Generator objective weights matching the released training recipe."""

    adversarial: float = 1.0
    reconstruction: float = 100.0
    dice: float = 0.0
    structural_similarity: float = 50.0
    edge: float = 10.0


class PalmGANTrainer:
    """Own the PALM-GAN models, optimizers, and alternating LSGAN update.

    Inputs and targets must be grayscale float tensors in ``[0, 1]`` with
    shape ``[batch, 1, height, width]``. The target convention is chosen by the
    dataset: use clean grayscale targets for restoration or binary masks for
    document binarization, but do not mix conventions within one run.
    """

    def __init__(
        self,
        generator: PalmGANGenerator | None = None,
        discriminator: PalmGANPatchDiscriminator | None = None,
        *,
        generator_learning_rate: float = 2e-4,
        discriminator_learning_rate: float = 2e-4,
        betas: tuple[float, float] = (0.5, 0.999),
        loss_weights: LossWeights | None = None,
        adversarial_start_step: int = 0,
        gradient_clip_norm: float | None = 5.0,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.generator = (generator or PalmGANGenerator()).to(self.device)
        self.discriminator = (discriminator or PalmGANPatchDiscriminator()).to(
            self.device
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=generator_learning_rate, betas=betas
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_learning_rate,
            betas=betas,
        )
        self.loss_weights = loss_weights or LossWeights()
        self.adversarial_start_step = adversarial_start_step
        self.gradient_clip_norm = gradient_clip_norm
        self.step = 0

    @staticmethod
    def _validate_batch(degraded: torch.Tensor, clean: torch.Tensor) -> None:
        if degraded.shape != clean.shape:
            raise ValueError("degraded and clean batches must have identical shapes")
        if degraded.ndim != 4 or degraded.shape[1] != 1:
            raise ValueError("batches must be shaped [batch, 1, height, width]")

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def train_step(
        self, degraded: torch.Tensor, clean: torch.Tensor
    ) -> dict[str, float]:
        self._validate_batch(degraded, clean)
        degraded = degraded.to(self.device, non_blocking=True)
        clean = clean.to(self.device, non_blocking=True)
        use_adversarial = self.step >= self.adversarial_start_step

        self.generator.train()
        self.discriminator.train()
        discriminator_loss = torch.zeros((), device=self.device)

        if use_adversarial:
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                generated = torch.sigmoid(self.generator(degraded))
            real_logits = self.discriminator(degraded, clean)
            fake_logits = self.discriminator(degraded, generated)
            discriminator_loss = 0.5 * (
                (real_logits - 1.0).square().mean() + fake_logits.square().mean()
            )
            discriminator_loss.backward()
            self._clip(self.discriminator.parameters())
            self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad(set_to_none=True)
        generated_logits = self.generator(degraded)
        generated = torch.sigmoid(generated_logits)
        adversarial_loss = torch.zeros((), device=self.device)
        if use_adversarial:
            self._set_requires_grad(self.discriminator, False)
            self.discriminator.eval()
            adversarial_loss = (
                0.5 * (self.discriminator(degraded, generated) - 1.0).square().mean()
            )

        reconstruction_loss = F.binary_cross_entropy_with_logits(
            generated_logits, clean
        )
        overlap_loss = dice_loss(generated, clean)
        ssim_loss = structural_similarity_loss(generated, clean)
        edge_loss = edge_consistency_loss(generated, clean)
        weights = self.loss_weights
        generator_loss = (
            weights.adversarial * adversarial_loss
            + weights.reconstruction * reconstruction_loss
            + weights.dice * overlap_loss
            + weights.structural_similarity * ssim_loss
            + weights.edge * edge_loss
        )
        generator_loss.backward()
        self._clip(self.generator.parameters())
        self.generator_optimizer.step()

        if use_adversarial:
            self._set_requires_grad(self.discriminator, True)
            self.discriminator.train()
        self.step += 1
        return {
            "generator_loss": float(generator_loss.detach()),
            "discriminator_loss": float(discriminator_loss.detach()),
            "adversarial_loss": float(adversarial_loss.detach()),
            "reconstruction_loss": float(reconstruction_loss.detach()),
            "dice_loss": float(overlap_loss.detach()),
            "ssim_loss": float(ssim_loss.detach()),
            "edge_loss": float(edge_loss.detach()),
        }

    def _clip(self, parameters: Any) -> None:
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)

    @torch.no_grad()
    def predict(self, degraded: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        return torch.sigmoid(self.generator(degraded.to(self.device)))

    def save_checkpoint(self, path: str | Path, **metadata: Any) -> None:
        """Save a resumable checkpoint with JSON-compatible metadata."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                "loss_weights": asdict(self.loss_weights),
                "step": self.step,
                "metadata": metadata,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer"]
        )
        self.step = int(checkpoint["step"])
        return dict(checkpoint.get("metadata", {}))
