# 📜 Generate, Transform, and Clean: The Role of GANs and Transformers in Palm Leaf Manuscript Generation and Enhancement [Updated]

This repository accompanies the paper:

**"Generate, transform, and clean: the role of GANs and transformers in palm leaf manuscript generation and enhancement"**

---
Progress:
1- Paper Accepted
2- Release Code (Update)

## 🎯 Research Goals

This paper investigates how recent advances in deep learning—specifically Generative Adversarial Networks (GANs) and Transformer-based models—can be leveraged to:

1. **Generate** synthetic palm leaf manuscripts for augmenting limited datasets and simulating degradation.  
2. **Transform** degraded manuscript images into enhanced, readable versions.  
3. **Clean** noise, stains, and artifacts to facilitate automated analysis and digital archiving.  

---

## 🔑 Key Contributions

- **GAN-based Generation:**  
  GANs are trained to synthesize realistic palm leaf manuscript images, expanding training datasets and simulating degradation for robust model development.

- **Transformer-based Enhancement:**  
  Vision Transformers and related architectures restore fine script details by tackling denoising, inpainting, and super-resolution, achieving state-of-the-art performance.

- **Hybrid Restoration Pipelines:**  
  Combines the strengths of GANs (generation & inpainting) with transformers (contextual enhancement & cleaning), showing superior results compared to single-model approaches.

- **Evaluation & Benchmarks:**  
  Provides both quantitative metrics (PSNR, SSIM) and qualitative results on real and synthetic datasets, demonstrating significant improvements in readability and visual quality.

---

## Memory-safe PyTorch implementation

The `palmgan` package provides the paper-aligned residual CNN encoder/decoder,
Transformer context blocks, and conditional PatchGAN discriminator. Local
window attention is used at high-resolution stages to avoid the quadratic
memory cost of full 256 x 256 attention. The training helper implements LSGAN,
binary reconstruction, Dice, SSIM, and edge-consistency objectives.

```bash
python -m pip install -e ".[test]"
pytest -q
```

```python
import torch

from palmgan import PalmGANGenerator, PalmGANTrainer

model = PalmGANGenerator()
degraded = torch.rand(1, 1, 256, 256)
clean_probability = torch.sigmoid(model(degraded))

trainer = PalmGANTrainer(generator=model, adversarial_start_step=500)
metrics = trainer.train_step(degraded, torch.rand_like(degraded))
```

`configs/train_palmgan_v2.yaml` documents the recommended starting recipe.
Use image-level train/validation splits and tune the final binarization
threshold on validation data only.

---

## 📖 Citation

If you use this code, data, or ideas from our work, please cite:

```bibtex
@article{thuon2024generate,
  title={Generate, transform, and clean: the role of GANs and transformers in palm leaf manuscript generation and enhancement},
  author={Thuon, Nimol and Du, Jun and Zhang, Zhenrong and Ma, Jiefeng and Hu, Pengfei},
  journal={International Journal on Document Analysis and Recognition (IJDAR)},
  volume={27},
  number={3},
  pages={415--432},
  year={2024},
  publisher={Springer}
}
```
