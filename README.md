# PALM-GAN

**A memory-efficient CNN–Transformer conditional GAN for palm-leaf manuscript
enhancement and document binarization.**

[Paper](https://doi.org/10.1007/s10032-024-00472-z) ·
[Project page](https://ruisju111.github.io/enhancement/) ·
[Configuration](configs/train_palmgan_v2.yaml) ·
[Tests](tests/test_palmgan.py)

PALM-GAN is the cleaning component of the **Generate–Transform–Clean (GTC)**
framework introduced in *Generate, transform, and clean: the role of GANs and
transformers in palm leaf manuscript generation and enhancement*. The framework
studies synthetic manuscript generation, degradation transfer, and restoration
for Balinese, Sundanese, and Khmer palm-leaf manuscripts.

The current PyTorch implementation combines a residual convolutional
encoder–decoder with Transformer context blocks and a conditional PatchGAN
discriminator. It follows the released PALM-GAN design while replacing
expensive high-resolution global attention with bounded local-window attention.


## Highlights

- Residual CNN encoder–decoder with multi-scale skip connections.
- Transformer context modeling at encoder, bottleneck, and decoder stages.
- Windowed high-resolution attention with global bottleneck attention.
- Conditional PatchGAN discriminator with optional spectral normalization.
- Alternating least-squares GAN (LSGAN) optimization.
- BCE reconstruction, Dice, SSIM, and edge-consistency objectives.
- Configurable adversarial warm-up and gradient clipping.
- Resumable generator, discriminator, and optimizer checkpoints.

PALM-GAN is inspired by conditional document-enhancement models such as
[DE-GAN](https://arxiv.org/abs/2010.08764), with Transformer layers added to
capture longer-range manuscript structure.

## Model overview

```text
degraded manuscript
        │
        ▼
residual CNN encoder ──────┐
        │                  │ skip connections
window/global Transformer │
        │                  │
        ▼                  │
residual CNN decoder ◄─────┘
        │
        ▼
clean-image logits ── sigmoid ── clean probability map

PatchGAN discriminator: D(degraded image, real or generated clean image)
```

The generator accepts grayscale tensors shaped `[batch, 1, height, width]` and
returns logits with the same spatial dimensions. Inputs and targets must be
normalized to `[0, 1]`. Use one target convention throughout an experiment:
either clean grayscale images for restoration or binary masks for binarization.

## Installation

Requirements:

- Python 3.10 or newer
- PyTorch 2.2 or newer

```bash
git clone https://github.com/back-kh/PALM-GAN.git
cd PALM-GAN

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
pytest -q
```

`PalmGANTrainer` selects CUDA automatically when it is available. You can pass
`device="cpu"`, `device="cuda"`, or another PyTorch device explicitly.

## Inference

```python
import torch

from palmgan import PalmGANGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PalmGANGenerator().to(device).eval()

# Replace this tensor with a normalized grayscale manuscript batch.
degraded = torch.rand(1, 1, 256, 256, device=device)

with torch.inference_mode():
    clean_probability = torch.sigmoid(model(degraded))
    binary_output = clean_probability >= 0.5
```

The threshold `0.5` is only a starting point. Select the final binarization
threshold on validation data and freeze it before evaluating the test set.

## Training

`PalmGANTrainer` performs one alternating discriminator/generator update for
each call to `train_step`:

```python
from palmgan import LossWeights, PalmGANGenerator, PalmGANTrainer

trainer = PalmGANTrainer(
    generator=PalmGANGenerator(),
    generator_learning_rate=2e-4,
    discriminator_learning_rate=2e-4,
    betas=(0.5, 0.999),
    adversarial_start_step=500,
    loss_weights=LossWeights(
        adversarial=1.0,
        reconstruction=100.0,
        dice=0.0,
        structural_similarity=50.0,
        edge=10.0,
    ),
)

for degraded, clean_target in train_loader:
    metrics = trainer.train_step(degraded, clean_target)

trainer.save_checkpoint(
    "checkpoints/palmgan.pt",
    epoch=1,
    target_convention="clean-background",
)
```

The repository intentionally leaves dataset loading to the experiment code
because manuscript collections use different directory structures and
annotation conventions. A batch supplied by a data loader must contain paired,
spatially aligned degraded and clean tensors.

The values in
[`configs/train_palmgan_v2.yaml`](configs/train_palmgan_v2.yaml) are a starting
recipe, not guaranteed optimal hyperparameters.


## Evaluation and reproducibility

For binarization experiments, consider reporting F-measure, pseudo-F-measure,
PSNR, and distance reciprocal distortion (DRD). For grayscale restoration,
report appropriate image-quality metrics such as PSNR and SSIM alongside
qualitative manuscript examples.

Each experiment should record:

- Dataset version and document-level split identifiers.
- Pretraining source and checkpoint provenance.
- Image normalization and target convention.
- Patch sampling and augmentation configuration.
- Random seeds, software versions, and hardware.
- Epoch-selection and early-stopping rules.
- Validation-selected threshold and untouched test results.

Do not treat unit tests, smoke tests, training loss, or validation-only
measurements as final benchmark results.

## Repository layout

| Path | Purpose |
| --- | --- |
| `palmgan/model.py` | Model architecture and discriminator |
| `palmgan/losses.py` | Dice, SSIM, and edge-consistency losses |
| `palmgan/training.py` | LSGAN updates, prediction, and checkpoints |
| `configs/train_palmgan_v2.yaml` | Reference hyperparameters |
| `tests/test_palmgan.py` | Model and training tests |
| `generate/` | Research-release generation scripts |
| `clean/` and root TensorFlow scripts | Research-release cleaning code |
| `data/` | Original data-preparation scripts |
| `Syllable Analysis/` | Language analysis resources |

The `palmgan/` package is the supported installable PyTorch implementation. The
original TensorFlow and research-pipeline files are retained for provenance and
may require environment-specific paths or adaptation.

## Development

Before opening a pull request, run:

```bash
pytest -q
ruff format --check palmgan tests
ruff check palmgan tests
```

Keep code changes separate from datasets, checkpoints, generated samples, and
benchmark outputs. Large research artifacts should be published through a
versioned release or an archival data service with checksums and provenance.

## Citation

If you use the paper, code, or GTC methodology, please cite:

```bibtex
@article{thuon2024generate,
  title   = {Generate, transform, and clean: the role of GANs and transformers
             in palm leaf manuscript generation and enhancement},
  author  = {Thuon, Nimol and Du, Jun and Zhang, Zhenrong and Ma, Jiefeng and
             Hu, Pengfei},
  journal = {International Journal on Document Analysis and Recognition},
  volume  = {27},
  number  = {3},
  pages   = {415--432},
  year    = {2024},
  doi     = {10.1007/s10032-024-00472-z}
}
```

## License

This repository does not currently contain a license file. Unless a license is
added, copyright remains with the respective authors; contact the maintainers
before redistribution or incorporation into another project.
