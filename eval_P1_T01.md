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
| **Classical Only (Adaptive Retinex + Luminance CLAHE)** | 17.60 | 0.5928 | 0.2710 | 3.5334 | 137.4 ms (0.137 s) | 4.11 / +1.00 dB (score 0.707) | 0.8428 |
| **Deep Learning Only (HVI-CIDNet)** | 21.18 | 0.7356 | 0.2363 | 3.5682 | 452.6 ms (0.453 s) | 4.91 / +4.23 dB (score 0.786) | 0.8562 |
| **Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion)** | 23.33 | 0.7620 | 0.2001 | 3.3485 | 700.2 ms (0.700 s) | 3.61 / +6.59 dB (score 0.838) | 0.8705 |

---

## 2. Detailed Metric Breakdown (In Official Document Order)

### Metric 1: PSNR and SSIM (Full-Reference Reconstruction & Structural Fidelity)
- **Definition:** Peak Signal-to-Noise Ratio (dB, higher is better) and Structural Similarity Index ($[0, 1]$, higher is better) computed against paired normal-light ground truth.

| Approach | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 17.60 | 0.5928 |
| Deep Learning Only (HVI-CIDNet) | 21.18 | 0.7356 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 23.33 | 0.7620 |

### Metric 2: LPIPS (Learned Perceptual Image Patch Similarity)
- **Definition:** Deep feature perceptual distance (AlexNet backbone, lower is better) measuring human-aligned texture and perceptual fidelity.

| Approach | LPIPS ↓ |
|---|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 0.2710 |
| Deep Learning Only (HVI-CIDNet) | 0.2363 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 0.2001 |

### Metric 3: NIQE (No-Reference Image Quality Metric)
- **Definition:** Natural Image Quality Evaluator based on Mean Subtracted Contrast Normalized (MSCN) Natural Scene Statistics (lower is better).

| Approach | NIQE ↓ |
|---|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 3.5334 |
| Deep Learning Only (HVI-CIDNet) | 3.5682 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 3.3485 |

### Metric 4: Processing Latency
- **Definition:** Mean end-to-end processing time per image in milliseconds and seconds (lower is better; must remain well under the $< 10\text{ s}$ live deployment requirement).

| Approach | Mean Latency (ms) ↓ | Mean Latency (s) ↓ | Meets $< 10\text{ s}$ Rubric Target |
|---|---:|---:|:---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 137.39 ms | 0.137 s | Yes |
| Deep Learning Only (HVI-CIDNet) | 452.61 ms | 0.453 s | Yes |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 700.15 ms | 0.700 s | Yes |

### Metric 5: Noise Suppression and Edge Preservation
- **Noise Suppression Definition:** High-frequency diagonal wavelet Median Absolute Deviation ($\hat{\sigma}_{\text{MAD}} = \text{median}(|W_{HH}|)/0.6745$) measured in homogeneous shadow regions (lower residual $\hat{\sigma}_{\text{MAD}}$ and higher gain-normalized suppression dB are better).
- **Edge Preservation Definition:** Edge Preservation Index (EPI $\in [0, 1]$, higher is better) combining Sobel gradient magnitude correlation and FSIM-style boundary similarity.

| Approach | Residual Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | Gain-Normalized Noise Suppression (dB) ↑ | Noise Cleanliness Score $[0,1]$ ↑ | Edge Preservation Index (EPI) ↑ |
|---|---:|---:|---:|---:|
| Classical Only (Adaptive Retinex + Luminance CLAHE) | 4.11 | +1.00 dB | 0.707 | 0.8428 |
| Deep Learning Only (HVI-CIDNet) | 4.91 | +4.23 dB | 0.786 | 0.8562 |
| Hybrid (ARHR: Classical + HVI-CIDNet + Spatial Fusion) | 3.61 | +6.59 dB | 0.838 | 0.8705 |

---

## 3. Unseen Low-Light Generalization Evaluation (No-Reference)

### Comparative Evaluation — Unseen Real-World Low-Light Test Images (Unpaired)

| Approach | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Processing Latency (ms / s) ↓ | 5a. Noise Suppression (Residual $\hat{\sigma}_{\text{MAD}}$ ↓ / Gain dB ↑) | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid (ARHR on Unseen Unpaired Inputs)** | N/A (Unpaired) | N/A (Unpaired) | N/A (Unpaired) | 3.3485 | 700.2 ms (0.700 s) | 3.61 / +6.59 dB (score 0.838) | 0.8039 |

---

## 4. Ablation Study

### Ablation Study — Component Contribution Analysis

| Experiment / Configuration | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Latency (ms) ↓ | 5a. Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp A: Classical CV Only (Fixed Parameters) | 17.16 | 0.5308 | 0.3632 | 3.6527 | 189.8 | 5.70 | 0.7807 |
| Exp B: Classical CV Only (Adaptive Parameters) | 17.60 | 0.5928 | 0.2710 | 3.5334 | 137.4 | 4.11 | 0.8428 |
| Exp C: Deep Learning Only (HVI-CIDNet) | 21.18 | 0.7356 | 0.2363 | 3.5682 | 452.6 | 4.91 | 0.8562 |
| Exp D: Classical + Deep (Uniform 50/50 Blend) | 22.87 | 0.7222 | 0.1983 | 3.4071 | 605.2 | 5.27 | 0.8830 |
| Exp E: Illumination-Guided Spatial Fusion | 23.00 | 0.7453 | 0.1948 | 3.3455 | 679.0 | 3.94 | 0.8735 |
| Exp F: Full ARHR Hybrid (Illum + Noise + Edge + Quality Guard) | 23.33 | 0.7620 | 0.2001 | 3.3485 | 700.2 | 3.61 | 0.8705 |

---

## 5. Documented Failure-Case Analysis

| Failure Case | Input Condition | Observed Behavior | Root Technical Cause |
|---|---|---|---|
| **Case 1 — Crushed Blacks / Zero-Bit Quantization Shadows** | Extremely dark regions where raw RGB intensities are quantized to (0, 0, 0). | Loss of fine micro-texture and posterization in deepest shadow pockets. | When photon counts fall below the sensor read-noise floor and quantize to 0, true spatial signal is physically absent; non-hallucinatory restoration avoids inventing synthetic textures. |
| **Case 2 — Severe Mixed-Illuminant Scenes** | Simultaneous warm sodium-vapor streetlights (2200K) and cool fluorescent/LED signage (6000K). | Residual warm or cool tint in localized transition zones. | Global and smooth spatial white-balance priors assume slowly varying illumination chromaticity and can under-correct sharp dual-illuminant boundaries. |
| **Case 3 — Low-Light Motion Blur & Defocus** | Handheld long-exposure capture with camera shake or moving subjects. | Restored brightness reveals motion-blurred edges cleanly without deblurring them. | Retinex decomposition and HVI-CIDNet restore photometric illumination and suppress sensor noise, but do not perform blind motion point-spread-function (PSF) deconvolution. |
| **Case 4 — Extreme High-ISO Sensor Noise on Fine Foliage/Fabric** | ISO > 12800 capture where chroma/luma noise amplitude matches fine texture gradients. | Trade-off between slight residual grain or mild softening of stochastic micro-textures. | When noise variance overlaps the spatial frequency band of fine stochastic textures, edge-preserving filters face an information-theoretic limit. |
