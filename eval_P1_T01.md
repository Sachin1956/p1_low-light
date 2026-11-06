1RV23CS001, 1RV23CS002, 1RV23CS003, 1RV23CS004

# Comparative Evaluation Results — Problem Statement 1 (`eval_P1_T01`)

## Evaluation Protocol & Environment
- **Problem ID:** P1 — Robust Low-Light Image Restoration and Enhancement
- **Dataset & Split:** Paired Low-Light Verification Benchmark (4 scenes) (4 images evaluated)
- **Input Resolution:** 600x400
- **Hardware:** CPU
- **Timing Protocol:** End-to-end inference time per image (including image characteristic analysis, restoration, spatial fusion, and quality-aware refinement).

---

## 1. Comparative Evaluation Summary (Exact P1 Metric Order)

### Comparative Evaluation — Paired Low-Light Verification Benchmark (4 scenes)

| Approach | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Processing Latency (ms / s) ↓ | 5a. Noise Suppression (Residual $\hat{\sigma}_{\text{MAD}}$ ↓ / Gain dB ↑) | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Classical Only (Adaptive Retinex + Luminance CLAHE)** | 16.14 | 0.6566 | 0.2426 | 6.3509 | 48.7 ms (0.049 s) | 2.68 / +4.54 dB (score 0.787) | 0.8859 |
| **Deep Learning Only (HVI-CIDNet)** | 21.38 | 0.7465 | 0.2458 | 5.6124 | 414.5 ms (0.415 s) | 4.24 / +5.09 dB (score 0.814) | 0.8549 |
| **Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion)** | 23.86 | 0.7778 | 0.2154 | 5.2744 | 543.8 ms (0.544 s) | 2.83 / +8.71 dB (score 0.872) | 0.8697 |

---

## 2. Detailed Metric Breakdown (In Official Document Order)

### Metric 1: PSNR and SSIM (Full-Reference Reconstruction & Structural Fidelity)
- **Definition:** Peak Signal-to-Noise Ratio (dB, higher is better) and Structural Similarity Index ($[0, 1]$, higher is better) computed against paired normal-light ground truth.

| Approach | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 16.14 | 0.6566 |
| Deep Learning Only (HVI-CIDNet) | 21.38 | 0.7465 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 23.86 | 0.7778 |

### Metric 2: LPIPS (Learned Perceptual Image Patch Similarity)
- **Definition:** Deep feature perceptual distance (AlexNet backbone, lower is better) measuring human-aligned texture and perceptual fidelity.

| Approach | LPIPS ↓ |
|---|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 0.2426 |
| Deep Learning Only (HVI-CIDNet) | 0.2458 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 0.2154 |

### Metric 3: NIQE (No-Reference Image Quality Metric)
- **Definition:** Natural Image Quality Evaluator based on Mean Subtracted Contrast Normalized (MSCN) Natural Scene Statistics (lower is better).

| Approach | NIQE ↓ |
|---|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 6.3509 |
| Deep Learning Only (HVI-CIDNet) | 5.6124 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 5.2744 |

### Metric 4: Processing Latency
- **Definition:** Mean end-to-end processing time per image in milliseconds and seconds (lower is better; must remain well under the $< 10\text{ s}$ live deployment requirement).

| Approach | Mean Latency (ms) ↓ | Mean Latency (s) ↓ | Meets $< 10\text{ s}$ Rubric Target |
|---|---:|---:|:---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 48.74 ms | 0.049 s | Yes |
| Deep Learning Only (HVI-CIDNet) | 414.51 ms | 0.415 s | Yes |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 543.78 ms | 0.544 s | Yes |

### Metric 5: Noise Suppression and Edge Preservation
- **Noise Suppression Definition:** High-frequency diagonal wavelet Median Absolute Deviation ($\hat{\sigma}_{\text{MAD}} = \text{median}(|W_{HH}|)/0.6745$) measured in homogeneous shadow regions (lower residual $\hat{\sigma}_{\text{MAD}}$ and higher gain-normalized suppression dB are better).
- **Edge Preservation Definition:** Edge Preservation Index (EPI $\in [0, 1]$, higher is better) combining Sobel gradient magnitude correlation and FSIM-style boundary similarity.

| Approach | Residual Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | Gain-Normalized Noise Suppression (dB) ↑ | Noise Cleanliness Score $[0,1]$ ↑ | Edge Preservation Index (EPI) ↑ |
|---|---:|---:|---:|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 2.68 | +4.54 dB | 0.787 | 0.8859 |
| Deep Learning Only (HVI-CIDNet) | 4.24 | +5.09 dB | 0.814 | 0.8549 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 2.83 | +8.71 dB | 0.872 | 0.8697 |

---

## 3. Unseen Low-Light Generalization Evaluation (No-Reference)

### Comparative Evaluation — Unseen Real-World Low-Light Test Images (Unpaired)

| Approach | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Processing Latency (ms / s) ↓ | 5a. Noise Suppression (Residual $\hat{\sigma}_{\text{MAD}}$ ↓ / Gain dB ↑) | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid (ARHR on Unseen Unpaired Inputs)** | N/A (Unpaired) | N/A (Unpaired) | N/A (Unpaired) | 5.2744 | 543.8 ms (0.544 s) | 2.83 / +8.71 dB (score 0.872) | 0.7916 |

---

## 4. Ablation Study

### Ablation Study — Component Contribution Analysis

| Experiment / Configuration | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Latency (ms) ↓ | 5a. Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp A: Classical CV Only (Fixed Parameters) | 18.66 | 0.5962 | 0.3099 | 7.2449 | 89.7 | 4.99 | 0.8380 |
| Exp B: Classical CV Only (Adaptive Parameters) | 16.14 | 0.6566 | 0.2426 | 6.3509 | 48.7 | 2.68 | 0.8859 |
| Exp C: Deep Learning Only (HVI-CIDNet) | 21.38 | 0.7465 | 0.2458 | 5.6124 | 414.5 | 4.24 | 0.8549 |
| Exp D: Classical + Deep (Uniform 50/50 Blend) | 22.26 | 0.7566 | 0.1865 | 6.2074 | 473.0 | 4.51 | 0.8982 |
| Exp E: Illumination-Guided Spatial Fusion | 23.17 | 0.7737 | 0.1956 | 5.3116 | 529.1 | 2.99 | 0.8759 |
| Exp F: Full ARHR Hybrid (Illum + Noise + Edge + Quality Guard) | 23.86 | 0.7778 | 0.2154 | 5.2744 | 543.8 | 2.83 | 0.8697 |

---

## 5. Documented Failure-Case Analysis

| Failure Case | Input Condition | Observed Behavior | Root Technical Cause |
|---|---|---|---|
| **Case 1 — Crushed Blacks / Zero-Bit Quantization Shadows** | Extremely dark regions where raw RGB intensities are quantized to (0, 0, 0). | Loss of fine micro-texture and posterization in deepest shadow pockets. | When photon counts fall below the sensor read-noise floor and quantize to 0, true spatial signal is physically absent; non-hallucinatory restoration avoids inventing synthetic textures. |
| **Case 2 — Severe Mixed-Illuminant Scenes** | Simultaneous warm sodium-vapor streetlights (2200K) and cool fluorescent/LED signage (6000K). | Residual warm or cool tint in localized transition zones. | Global and smooth spatial white-balance priors assume slowly varying illumination chromaticity and can under-correct sharp dual-illuminant boundaries. |
| **Case 3 — Low-Light Motion Blur & Defocus** | Handheld long-exposure capture with camera shake or moving subjects. | Restored brightness reveals motion-blurred edges cleanly without deblurring them. | Retinex decomposition and HVI-CIDNet restore photometric illumination and suppress sensor noise, but do not perform blind motion point-spread-function (PSF) deconvolution. |
| **Case 4 — Extreme High-ISO Sensor Noise on Fine Foliage/Fabric** | ISO > 12800 capture where chroma/luma noise amplitude matches fine texture gradients. | Trade-off between slight residual grain or mild softening of stochastic micro-textures. | When noise variance overlaps the spatial frequency band of fine stochastic textures, edge-preserving filters face an information-theoretic limit. |
