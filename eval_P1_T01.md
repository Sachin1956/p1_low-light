1RUA24CSE0375, 1RUA24CSE0376, 1RUA24CSE0384, 1RUA24CSE0395

# Comparative Evaluation Results — Problem Statement 1 (`eval_P1_T01`)

## Team Members
| Name | USN |
|---|---|
| Rohit Ajit Bharadwaj | `1RUA24CSE0375` |
| Rohith P Hegde | `1RUA24CSE0376` |
| S Sachithananthan | `1RUA24CSE0384` |
| Saket M | `1RUA24CSE0395` |

## Evaluation Protocol & Environment
- **Problem ID:** P1 — Robust Low-Light Image Restoration and Enhancement
- **Dataset & Split:** LOL-v2 Real (Validation Split) (5 images evaluated)
- **Input Resolution:** 600x400
- **Hardware:** CPU
- **Timing Protocol:** End-to-end inference time per image (including image characteristic analysis, restoration, spatial fusion, and quality-aware refinement).

---

## 1. Comparative Evaluation Summary (Exact P1 Metric Order)

### Comparative Evaluation — LOL-v2 Real (Validation Split)

| Approach | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Processing Latency (ms / s) ↓ | 5a. Noise Suppression (Residual $\hat{\sigma}_{\text{MAD}}$ ↓ / Gain dB ↑) | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Classical CV Only** | 14.74 | 0.4447 | 0.4999 | 9.0019 | 198.1 ms (0.198 s) | 10.30 / +-4.48 dB (score 0.884) | 0.7084 |
| **Deep Learning Only (Adapted)** | 26.49 | 0.8906 | 0.0710 | 4.8126 | 1730.5 ms (1.730 s) | 1.17 / +18.86 dB (score 0.996) | 0.9334 |
| **Adaptive Hybrid (ARHR Proposed)** | 21.18 | 0.7907 | 0.1665 | 6.5599 | 1839.9 ms (1.840 s) | 3.35 / +8.31 dB (score 0.984) | 0.8992 |

---

## 2. Detailed Metric Breakdown (In Official Document Order)

### Metric 1: PSNR and SSIM (Full-Reference Reconstruction & Structural Fidelity)
- **Definition:** Peak Signal-to-Noise Ratio (dB, higher is better) and Structural Similarity Index ($[0, 1]$, higher is better) computed against paired normal-light ground truth.

| Approach | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|
| Classical CV Only | 14.74 | 0.4447 |
| Deep Learning Only (Adapted) | 26.49 | 0.8906 |
| Adaptive Hybrid (ARHR Proposed) | 21.18 | 0.7907 |

### Metric 2: LPIPS (Learned Perceptual Image Patch Similarity)
- **Definition:** Deep feature perceptual distance (AlexNet backbone, lower is better) measuring human-aligned texture and perceptual fidelity.

| Approach | LPIPS ↓ |
|---|---:|
| Classical CV Only | 0.4999 |
| Deep Learning Only (Adapted) | 0.0710 |
| Adaptive Hybrid (ARHR Proposed) | 0.1665 |

### Metric 3: NIQE (No-Reference Image Quality Metric)
- **Definition:** Natural Image Quality Evaluator based on Mean Subtracted Contrast Normalized (MSCN) Natural Scene Statistics (lower is better).

| Approach | NIQE ↓ |
|---|---:|
| Classical CV Only | 9.0019 |
| Deep Learning Only (Adapted) | 4.8126 |
| Adaptive Hybrid (ARHR Proposed) | 6.5599 |

### Metric 4: Processing Latency
- **Definition:** Mean end-to-end processing time per image in milliseconds and seconds (lower is better; must remain well under the $< 10\text{ s}$ live deployment requirement).

| Approach | Mean Latency (ms) ↓ | Mean Latency (s) ↓ | Meets $< 10\text{ s}$ Rubric Target |
|---|---:|---:|:---:|
| Classical CV Only | 198.10 ms | 0.198 s | Yes |
| Deep Learning Only (Adapted) | 1730.50 ms | 1.730 s | Yes |
| Adaptive Hybrid (ARHR Proposed) | 1839.90 ms | 1.840 s | Yes |

### Metric 5: Noise Suppression and Edge Preservation
- **Noise Suppression Definition:** High-frequency diagonal wavelet Median Absolute Deviation ($\hat{\sigma}_{\text{MAD}} = \text{median}(|W_{HH}|)/0.6745$) measured in homogeneous shadow regions (lower residual $\hat{\sigma}_{\text{MAD}}$ and higher gain-normalized suppression dB are better).
- **Edge Preservation Definition:** Edge Preservation Index (EPI $\in [0, 1]$, higher is better) combining Sobel gradient magnitude correlation and FSIM-style boundary similarity.

| Approach | Residual Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | Gain-Normalized Noise Suppression (dB) ↑ | Noise Cleanliness Score $[0,1]$ ↑ | Edge Preservation Index (EPI) ↑ |
|---|---:|---:|---:|---:|
| Classical CV Only | 10.30 | +-4.48 dB | 0.884 | 0.7084 |
| Deep Learning Only (Adapted) | 1.17 | +18.86 dB | 0.996 | 0.9334 |
| Adaptive Hybrid (ARHR Proposed) | 3.35 | +8.31 dB | 0.984 | 0.8992 |

---

## 4. Ablation Study

### Ablation Study — Component Contribution Analysis

| Experiment / Configuration | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Latency (ms) ↓ | 5a. Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0: Fixed Gamma (0.5) | 9.72 | 0.4411 | 0.3855 | 7.6848 | 10.4 | 3.89 | 0.8100 |
| E1: Fixed Classical (Non-Adaptive) | 12.07 | 0.5071 | 0.4163 | 8.1166 | 70.5 | 4.43 | 0.7942 |
| E2: Adaptive Classical | 14.74 | 0.4447 | 0.4999 | 9.0019 | 75.0 | 10.30 | 0.7084 |
| E3: Pretrained Deep Backbone | 27.67 | 0.8824 | 0.0636 | 4.8474 | 1687.0 | 1.42 | 0.9338 |
| E4: Fine-Tuned Deep Backbone | 26.49 | 0.8906 | 0.0710 | 4.8126 | 1643.2 | 1.17 | 0.9334 |
| E5: 50/50 Uniform Blend | 19.62 | 0.6536 | 0.2988 | 7.7531 | 1991.2 | 6.03 | 0.8335 |
| E6: Illumination-Guided Fusion | 23.54 | 0.8486 | 0.1053 | 5.6927 | 1804.0 | 2.45 | 0.9245 |
| E7: Guided Fusion (Illum+Noise+Edge) | 21.18 | 0.7907 | 0.1665 | 6.5599 | 1814.5 | 3.35 | 0.8992 |
| E8: Guided Fusion + Quality Guard | 21.18 | 0.7907 | 0.1665 | 6.5599 | 1773.2 | 3.35 | 0.8992 |

---

## 5. Documented Failure-Case Analysis

| Failure Case | Input Condition | Observed Behavior | Root Technical Cause |
|---|---|---|---|
| **Case 1 — Crushed Blacks / Zero-Bit Quantization Shadows** | Extremely dark regions where raw RGB intensities are quantized to (0, 0, 0). | Loss of fine micro-texture and posterization in deepest shadow pockets. | When photon counts fall below the sensor read-noise floor and quantize to 0, true spatial signal is physically absent; non-hallucinatory restoration avoids inventing synthetic textures. |
| **Case 2 — Severe Mixed-Illuminant Scenes** | Simultaneous warm sodium-vapor streetlights (2200K) and cool fluorescent/LED signage (6000K). | Residual warm or cool tint in localized transition zones. | Global and smooth spatial white-balance priors assume slowly varying illumination chromaticity and can under-correct sharp dual-illuminant boundaries. |
| **Case 3 — Low-Light Motion Blur & Defocus** | Handheld long-exposure capture with camera shake or moving subjects. | Restored brightness reveals motion-blurred edges cleanly without deblurring them. | Retinex decomposition and HVI-CIDNet restore photometric illumination and suppress sensor noise, but do not perform blind motion point-spread-function (PSF) deconvolution. |
| **Case 4 — Extreme High-ISO Sensor Noise on Fine Foliage/Fabric** | ISO > 12800 capture where chroma/luma noise amplitude matches fine texture gradients. | Trade-off between slight residual grain or mild softening of stochastic micro-textures. | When noise variance overlaps the spatial frequency band of fine stochastic textures, edge-preserving filters face an information-theoretic limit. |
