import torch

from palmgan.losses import edge_consistency_loss, structural_similarity_loss
from palmgan.model import (
    PalmGANGenerator,
    PalmGANPatchDiscriminator,
    WindowTransformer2d,
)
from palmgan.training import LossWeights, PalmGANTrainer


def test_window_attention_preserves_non_multiple_shape() -> None:
    layer = WindowTransformer2d(16, heads=4, window_size=8, dropout=0.0)
    inputs = torch.randn(2, 16, 13, 19)
    outputs = layer(inputs)
    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()


def test_generator_and_discriminator_shapes() -> None:
    generator = PalmGANGenerator(base_channels=4).eval()
    discriminator = PalmGANPatchDiscriminator(
        base_channels=4, spectral_normalization=False
    ).eval()
    degraded = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        generated = torch.sigmoid(generator(degraded))
        decision = discriminator(degraded, generated)
    assert generated.shape == degraded.shape
    assert decision.shape == (2, 1, 4, 4)
    assert torch.isfinite(generated).all()
    assert torch.isfinite(decision).all()


def test_similarity_and_edge_losses_are_zero_for_identical_images() -> None:
    image = torch.rand(2, 1, 32, 32)
    assert torch.allclose(structural_similarity_loss(image, image), torch.tensor(0.0))
    assert torch.allclose(edge_consistency_loss(image, image), torch.tensor(0.0))


def test_adversarial_train_step_updates_both_models() -> None:
    generator = PalmGANGenerator(base_channels=4)
    discriminator = PalmGANPatchDiscriminator(
        base_channels=4, spectral_normalization=False
    )
    trainer = PalmGANTrainer(
        generator,
        discriminator,
        loss_weights=LossWeights(
            adversarial=1.0,
            reconstruction=1.0,
            structural_similarity=0.1,
            edge=0.1,
        ),
        gradient_clip_norm=1.0,
        device="cpu",
    )
    generator_before = next(generator.parameters()).detach().clone()
    discriminator_before = next(discriminator.parameters()).detach().clone()
    metrics = trainer.train_step(torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32))
    assert not torch.equal(generator_before, next(generator.parameters()).detach())
    assert not torch.equal(
        discriminator_before, next(discriminator.parameters()).detach()
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
