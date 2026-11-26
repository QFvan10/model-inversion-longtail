# Mining Long-Tail Information with Residual Dynamic Compensation for Black-Box Model Inversion Attacks

Reference implementation of a black-box model inversion pipeline that mines the suppressed tail of the posterior before decoding.

## Method

**NEAL (Nonlinear Entropy-Augmentation Layer)** reshapes each retained Top-k score `v_j` with a shifted convex combination of exponential and logarithmic branches:

```
T_γ(v_j) = γ · exp(v_j) + (1 - γ) · log(v_j + ε) + C
```

with `ε = 1e-38`, `γ = 0.3` and `C = 50`. A per-sample floor alignment keeps the output nonnegative before the scores are scattered back to the full `K`-dimensional class space. The derivative `T'_γ(x) = γ e^x + (1 - γ) / (x + ε)` is strictly positive, so class ordering is preserved, and the sensitivity converges to the nonzero floor `ln(10)(1 - γ)` in the deep tail.

**RDC (Residual Dynamic Compensation)** models cross-coordinate dependencies over the full augmented vector and injects a gated residual before decoding:

```
r = W₂ · GELU(W₁ · LN_r(z) + b₁) + b₂        # Tail Mixer
g = σ(W_g · LN_g(z) + b_g)                    # Tail Gate
z' = z + r ⊙ g
```

The mixer couples all `K` coordinates through two dense projections, and the gate modulates the correction per coordinate. Both branches are zero-initialised, so the module starts as an identity mapping. The compensated vector `z'` is decoded by a five-layer transposed-convolution network into a `1 × 64 × 64` image.

## Contents

| File | Description |
|---|---|
| `train_inversion.py` | Training entry point for the inversion network |
| `model.py` | Victim classifiers and the NEAL + RDC inversion models |
| `curated_chestxray_data.py` | Curated Chest X-ray split construction and dataset |
| `data.py` | FaceScrub / Fashion-MNIST / CelebA datasets |
| `metrics.py` | PSNR and SSIM |
| `experiment_io.py` | Checkpoint and run-directory bookkeeping |
| `requirements.txt` | Python dependencies |
