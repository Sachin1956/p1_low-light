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
- **Dataset & Split:** Paired Low-Light Verification Benchmark (5 scenes) (5 images evaluated)
- **Input Resolution:** 600x400
- **Hardware:** CPU
- **Timing Protocol:** End-to-end inference time per image (including image characteristic analysis, restoration, spatial fusion, and quality-aware refinement).

---

## 1. Comparative Evaluation Summary (Exact P1 Metric Order)

### Comparative Evaluation — Paired Low-Light Verification Benchmark (5 scenes)

| Approach | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Processing Latency (ms / s) ↓ | 5a. Noise Suppression (Residual $\hat{\sigma}_{\text{MAD}}$ ↓ / Gain dB ↑) | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Classical Only (Adaptive Retinex + Luminance CLAHE)** | 18.51 | 0.6314 | 0.2730 | 3.4802 | 87.5 ms (0.088 s) | 4.26 / +1.33 dB (score 0.717) | 0.8421 |
| **Deep Learning Only (HVI-CIDNet)** | 12.25 | 0.5497 | 0.2927 | 4.1241 | 558.3 ms (0.558 s) | 2.06 / +1.47 dB (score 0.718) | 0.8461 |
| **Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion)** | 14.65 | 0.6331 | 0.2538 | 3.7674 | 734.9 ms (0.735 s) | 2.74 / +1.66 dB (score 0.723) | 0.8728 |

---

## 2. Detailed Metric Breakdown (In Official Document Order)

### Metric 1: PSNR and SSIM (Full-Reference Reconstruction & Structural Fidelity)
- **Definition:** Peak Signal-to-Noise Ratio (dB, higher is better) and Structural Similarity Index ($[0, 1]$, higher is better) computed against paired normal-light ground truth.

| Approach | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 18.51 | 0.6314 |
| Deep Learning Only (HVI-CIDNet) | 12.25 | 0.5497 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 14.65 | 0.6331 |

### Metric 2: LPIPS (Learned Perceptual Image Patch Similarity)
- **Definition:** Deep feature perceptual distance (AlexNet backbone, lower is better) measuring human-aligned texture and perceptual fidelity.

| Approach | LPIPS ↓ |
|---|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 0.2730 |
| Deep Learning Only (HVI-CIDNet) | 0.2927 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 0.2538 |

### Metric 3: NIQE (No-Reference Image Quality Metric)
- **Definition:** Natural Image Quality Evaluator based on Mean Subtracted Contrast Normalized (MSCN) Natural Scene Statistics (lower is better).

| Approach | NIQE ↓ |
|---|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 3.4802 |
| Deep Learning Only (HVI-CIDNet) | 4.1241 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 3.7674 |

### Metric 4: Processing Latency
- **Definition:** Mean end-to-end processing time per image in milliseconds and seconds (lower is better; must remain well under the $< 10\text{ s}$ live deployment requirement).

| Approach | Mean Latency (ms) ↓ | Mean Latency (s) ↓ | Meets $< 10\text{ s}$ Rubric Target |
|---|---:|---:|:---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 87.54 ms | 0.088 s | Yes |
| Deep Learning Only (HVI-CIDNet) | 558.35 ms | 0.558 s | Yes |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 734.90 ms | 0.735 s | Yes |

### Metric 5: Noise Suppression and Edge Preservation
- **Noise Suppression Definition:** High-frequency diagonal wavelet Median Absolute Deviation ($\hat{\sigma}_{\text{MAD}} = \text{median}(|W_{HH}|)/0.6745$) measured in homogeneous shadow regions (lower residual $\hat{\sigma}_{\text{MAD}}$ and higher gain-normalized suppression dB are better).
- **Edge Preservation Definition:** Edge Preservation Index (EPI $\in [0, 1]$, higher is better) combining Sobel gradient magnitude correlation and FSIM-style boundary similarity.

| Approach | Residual Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | Gain-Normalized Noise Suppression (dB) ↑ | Noise Cleanliness Score $[0,1]$ ↑ | Edge Preservation Index (EPI) ↑ |
|---|---:|---:|---:|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 4.26 | +1.33 dB | 0.717 | 0.8421 |
| Deep Learning Only (HVI-CIDNet) | 2.06 | +1.47 dB | 0.718 | 0.8461 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 2.74 | +1.66 dB | 0.723 | 0.8728 |

---

## 3. Unseen Low-Light Generalization Evaluation (No-Reference)

### Comparative Evaluation — Unseen Real-World Low-Light Test Images (Unpaired)

| Approach | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Processing Latency (ms / s) ↓ | 5a. Noise Suppression (Residual $\hat{\sigma}_{\text{MAD}}$ ↓ / Gain dB ↑) | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid (ARHR on Unseen Unpaired Inputs)** | N/A (Unpaired) | N/A (Unpaired) | N/A (Unpaired) | 3.7674 | 734.9 ms (0.735 s) | 2.74 / +1.66 dB (score 0.723) | 0.9368 |

---

## 4. Ablation Study

### Ablation Study — Component Contribution Analysis

| Experiment / Configuration | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Latency (ms) ↓ | 5a. Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp A: Classical CV Only (Fixed Parameters) | 17.85 | 0.5540 | 0.3680 | 3.5986 | 107.5 | 5.84 | 0.7801 |
| Exp B: Classical CV Only (Adaptive Parameters) | 18.51 | 0.6314 | 0.2730 | 3.4802 | 87.5 | 4.26 | 0.8421 |
| Exp C: Deep Learning Only (HVI-CIDNet) | 12.25 | 0.5497 | 0.2927 | 4.1241 | 558.3 | 2.06 | 0.8461 |
| Exp D: Classical + Deep (Uniform 50/50 Blend) | 15.22 | 0.6464 | 0.2513 | 3.7054 | 659.0 | 3.12 | 0.8756 |
| Exp E: Illumination-Guided Spatial Fusion | 14.17 | 0.6483 | 0.2340 | 3.7237 | 711.7 | 1.16 | 0.8693 |
| Exp F: Full ARHR Hybrid (Illum + Noise + Edge + Quality Guard) | 14.65 | 0.6331 | 0.2538 | 3.7674 | 734.9 | 2.74 | 0.8728 |

---

## 5. Documented Failure-Case Analysis

| Failure Case | Input Condition | Observed Behavior | Root Technical Cause |
|---|---|---|---|
| **Case 1 — Crushed Blacks / Zero-Bit Quantization Shadows** | Extremely dark regions where raw RGB intensities are quantized to (0, 0, 0). | Loss of fine micro-texture and posterization in deepest shadow pockets. | When photon counts fall below the sensor read-noise floor and quantize to 0, true spatial signal is physically absent; non-hallucinatory restoration avoids inventing synthetic textures. |
| **Case 2 — Severe Mixed-Illuminant Scenes** | Simultaneous warm sodium-vapor streetlights (2200K) and cool fluorescent/LED signage (6000K). | Residual warm or cool tint in localized transition zones. | Global and smooth spatial white-balance priors assume slowly varying illumination chromaticity and can under-correct sharp dual-illuminant boundaries. |
| **Case 3 — Low-Light Motion Blur & Defocus** | Handheld long-exposure capture with camera shake or moving subjects. | Restored brightness reveals motion-blurred edges cleanly without deblurring them. | Retinex decomposition and HVI-CIDNet restore photometric illumination and suppress sensor noise, but do not perform blind motion point-spread-function (PSF) deconvolution. |
| **Case 4 — Extreme High-ISO Sensor Noise on Fine Foliage/Fabric** | ISO > 12800 capture where chroma/luma noise amplitude matches fine texture gradients. | Trade-off between slight residual grain or mild softening of stochastic micro-textures. | When noise variance overlaps the spatial frequency band of fine stochastic textures, edge-preserving filters face an information-theoretic limit. |
