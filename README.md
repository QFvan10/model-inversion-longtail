# Mining Long-Tail Information with Residual Dynamic Compensation for Black-Box Model Inversion Attacks

Reference implementation of the inversion pipeline described in *"Mining Long-Tail Information with Residual Dynamic Compensation for Black-Box Model Inversion Attacks"* (IEEE Transactions on Information Forensics and Security, under review).

**Feifan Que, Rongke Liu, Youwen Zhu** — School of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics

## Overview

Concentrated Softmax posteriors hide the fine-grained tail responses that distinguish images sharing the same class, which limits the fidelity of score-based black-box model inversion. Static scalar mappings (log, power) reshape every coordinate independently, so they either lose resolution in the deep tail or distort the dominant class scale and destroy Top-k rank alignment.

This repository implements a two-stage answer to that problem, operating on the attacker-visible posterior vector before decoding:

- **NEAL — Nonlinear Entropy-Augmentation Layer**: a monotonic, rank-preserving blend of exponential and logarithmic branches with a constant offset and per-sample floor alignment.
- **RDC — Residual Dynamic Compensation**: a gated residual module that models cross-coordinate dependencies over the full augmented vector, separating intra-class representations ahead of the decoder.

Both Full Softmax (`k = K`) and index-aligned Top-k (`k < K`) score interfaces are supported.

## Method

Given a retained Top-k window `v` with original class indices `I`, NEAL applies

```
T_γ(v_j) = γ · exp(v_j) + (1 - γ) · log(v_j + ε) + C
```

with `ε = 1e-38`, `γ = 0.3` and `C = 50` by default, followed by a per-sample floor shift and zero-padded scatter back into the full `K`-dimensional class space. The first derivative `T'_γ(x) = γ e^x + (1 - γ) / (x + ε)` is strictly positive, so class ordering is preserved; as `x` decreases the sensitivity converges to the nonzero floor `ln(10)(1 - γ)`, avoiding the deep-tail collapse of power scaling.

RDC then computes a signed correction from the complete augmented vector and injects it through coordinate-wise gating:

```
r = W₂ · GELU(W₁ · LN_r(z) + b₁) + b₂        # Tail Mixer
g = σ(W_g · LN_g(z) + b_g)                    # Tail Gate
z' = z + r ⊙ g
```

The mixer couples all `K` coordinates through two dense projections, so each correction depends on the joint response pattern rather than a single coordinate. Gating starts at zero, making the module an identity mapping at initialization. The compensated vector `z'` is decoded by a five-layer transposed-convolution network into a `1 × 64 × 64` image.

## Repository contents

| File | Description |
|---|---|
| `train_inversion.py` | Training entry point for the inversion network |
| `model.py` | Victim classifiers and the NEAL + RDC inversion models |
| `curated_chestxray_data.py` | Curated Chest X-ray split construction and dataset |
| `data.py` | FaceScrub / Fashion-MNIST / CelebA datasets |
| `metrics.py` | PSNR and SSIM |
| `experiment_io.py` | Checkpoint and run-directory bookkeeping |

The repository covers the attack side only. Victim classifiers are expected as pre-trained checkpoints (see below); their training code is not included.

## Installation

```bash
pip install -r requirements.txt
```

The reference environment uses CUDA 12.1 builds (`torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`). On CPU-only machines plain `pip install torch==2.5.1 torchvision==0.20.1` is sufficient.

## Data

All inputs are `1 × 64 × 64` grayscale images. Datasets are not redistributed here.

- **Curated Chest X-ray** — from the Kaggle [Curated Chest X-Ray Image Dataset for COVID-19](https://www.kaggle.com/datasets/unaissait/curated-chest-xray-image-dataset-for-covid19): 9,208 images over four classes (`COVID-19` 1,281, `Normal` 3,270, `Pneumonia-Bacterial` 3,001, `Pneumonia-Viral` 1,656). Place the class folders under `data/Curated X-Ray Dataset/` and export a split audit with:

  ```bash
  python curated_chestxray_data.py --data-root ./data --split-seed 1
  ```

  This validates the directory layout, writes `data/curated_splits/*.csv`, and records the SHA-256 split fingerprint used for checkpoint validation.
- **Fashion-MNIST** — `data/Fashion-MNIST/`
- **FaceScrub** — `data/facescrub/`

Victim and auxiliary (attacker) pools are disjoint and equally sized: the 9,208 Chest X-ray images are stratified into 4,604 victim and 4,604 attacker images, and each pool is split 80/10/10 into train/val/test (3,683 / 461 / 460). The two pools are drawn with the same class proportions, so no class is missing from any split. Note that the Chest X-ray source data carries no patient identifiers, so the partition is image-level, not patient-level.

## Training

Victim checkpoints are resolved from `<classifier-output-root>/<dataset>/<model>/latest.txt`. Once a checkpoint exists:

```bash
python train_inversion.py \
  --dataset curated-chestxray \
  --data-root "./data/Curated X-Ray Dataset" \
  --model vgg16 \
  --method inversion \
  --transform hybrid \
  --truncation 4 \
  --epochs 200 \
  --patience 10 \
  --classifier-output-root out/classifier_runs \
  --output-root out/inversion_runs
```

`--method` selects the pipeline branch:

| Value | Meaning |
|---|---|
| `inversion` | NEAL + RDC (proposed) |
| `noise` | relative-L2 Gaussian perturbation + RDC |
| `dropping` | random coordinate dropping + RDC |
| `baseline` | log / power mapping, plain decoder, no RDC (`--transform log` or `power`) |

Useful flags: `--gamma` (exp/log mixture, default `0.3`), `--c` (offset, default `50`), `--activation {tanh,prelu,gelu}`, `--noise-ratio`, `--drop-rate`, `--split-seed`.

## Reported results

Across FaceScrub, Fashion-MNIST and Curated Chest X-ray on six victim architectures (CNN, VGG16, ResNet-18/34, MobileNetV2/V3), the framework improves over representative score-transformation baselines including Amplified-MIA in all 18 private target reconstruction settings, reducing MSE by up to **43.0%** on FaceScrub and **30.8%** on Chest X-ray, and improving SSIM by up to **65.8%** on Fashion-MNIST.

## Citation

```bibtex
@article{que2026longtail,
  title  = {Mining Long-Tail Information with Residual Dynamic Compensation
            for Black-Box Model Inversion Attacks},
  author = {Que, Feifan and Liu, Rongke and Zhu, Youwen},
  journal= {IEEE Transactions on Information Forensics and Security},
  year   = {2026},
  note   = {Under review}
}
```

## Contact

Feifan Que — sz2616170@nuaa.edu.cn
