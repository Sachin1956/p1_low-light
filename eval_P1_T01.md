1RUA24CSE0375, 1RUA24CSE0376, 1RUA24CSE0384, 1RUA24CSE0395

# Comparative Evaluation Results — Problem Statement 1 (`eval_P1_T01`)

The [submission benchmark](outputs/submission/summary.md) uses the deployed app settings on all 100 LOL-v2 Real test inputs and scores saved predictions in a separate pass. The tables below preserve the earlier development results and ablations. The legacy scorer permitted LPIPS/NIQE approximations, so use the new benchmark for standard metric comparisons. The [submission runbook](docs/P1_SUBMISSION_RUNBOOK.md) documents the current protocol and failure artifacts.

## Submission comparison · LOL-v2 Real official test

| Approach | 1. RGB PSNR ↑ | 1. SSIM ↑ | 2. LPIPS Alex ↓ | 3. NIQE ↓ | 4. Processing latency | 5a. Shadow MAD ↓ | 5b. Paired edge correlation ↑ |
|---|---:|---:|---:|---:|---|---:|---:|
| Classical | 15.9323 | 0.3951 | 0.4944 | 8.9771 | Shared request | 15.2261 | 0.7414 |
| Deep | 20.0330 | 0.8444 | 0.1559 | 4.4125 | Shared request | 1.0600 | 0.8765 |
| Adaptive hybrid | 21.5660 | 0.7623 | 0.1923 | 6.4309 | Shared request | 4.3143 | 0.8712 |

Hybrid PSNR improved by 1.53 dB over deep-only, with a paired ID bootstrap 95% interval of +0.87 to +2.19 dB. Deep-only leads on SSIM, LPIPS, NIQE, shadow noise and edge correlation. The full hybrid exceeds the plan's SSIM and LPIPS regression tolerances. The report therefore shows a PSNR tradeoff across metrics rather than a hybrid win on every gate. All 300 predictions were saved and hashed before the paired references were downloaded. Five test inputs previously appeared in the input-only checkpoint handoff.

One CPU request computes all three outputs and four maps. Native-size decode, processing and PNG writes took 1.95 s median, 4.36 s p95 and 5.95 s maximum. The separate public API audit completed 20 of 20 uploads and downloads, with 11.26 s median, 12.80 s p95 and 13.12 s maximum. Nine requests took less than 10 s. See [per-image scores](outputs/submission/scores.json), [prediction provenance](outputs/submission/inference.json), [public latency](outputs/submission/hosted_latency.json) and [inspected failure cases](outputs/submission/failures.md).

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
| **Classical CV Only** | 14.74 | 0.4447 | 0.4999 | 9.0019 | 247.9 ms (0.248 s) | 10.30 / +-4.48 dB (score 0.884) | 0.7084 |
| **Deep Learning Only (Adapted)** | 27.52 | 0.9002 | 0.0742 | 4.8376 | 2090.0 ms (2.090 s) | 0.80 / +22.44 dB (score 0.998) | 0.9338 |
| **Adaptive Hybrid (ARHR Proposed)** | 22.91 | 0.8062 | 0.1612 | 6.4326 | 2106.6 ms (2.107 s) | 3.24 / +9.32 dB (score 0.986) | 0.9004 |

---

## 2. Detailed Metric Breakdown (In Official Document Order)

### Metric 1: PSNR and SSIM (Full-Reference Reconstruction & Structural Fidelity)
- **Definition:** Peak Signal-to-Noise Ratio (dB, higher is better) and Structural Similarity Index ($[0, 1]$, higher is better) computed against paired normal-light ground truth.

| Approach | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|
| Classical CV Only | 14.74 | 0.4447 |
| Deep Learning Only (Adapted) | 27.52 | 0.9002 |
| Adaptive Hybrid (ARHR Proposed) | 22.91 | 0.8062 |

### Metric 2: LPIPS (Learned Perceptual Image Patch Similarity)
- **Definition:** Deep feature perceptual distance (AlexNet backbone, lower is better) measuring human-aligned texture and perceptual fidelity.

| Approach | LPIPS ↓ |
|---|---:|
| Classical CV Only | 0.4999 |
| Deep Learning Only (Adapted) | 0.0742 |
| Adaptive Hybrid (ARHR Proposed) | 0.1612 |

### Metric 3: NIQE (No-Reference Image Quality Metric)
- **Definition:** Natural Image Quality Evaluator based on Mean Subtracted Contrast Normalized (MSCN) Natural Scene Statistics (lower is better).

| Approach | NIQE ↓ |
|---|---:|
| Classical CV Only | 9.0019 |
| Deep Learning Only (Adapted) | 4.8376 |
| Adaptive Hybrid (ARHR Proposed) | 6.4326 |

### Metric 4: Processing Latency
- **Definition:** Mean end-to-end processing time per image in milliseconds and seconds (lower is better; must remain well under the $< 10\text{ s}$ live deployment requirement).

| Approach | Mean Latency (ms) ↓ | Mean Latency (s) ↓ | Meets $< 10\text{ s}$ Rubric Target |
|---|---:|---:|:---:|
| Classical CV Only | 247.90 ms | 0.248 s | Yes |
| Deep Learning Only (Adapted) | 2090.00 ms | 2.090 s | Yes |
| Adaptive Hybrid (ARHR Proposed) | 2106.60 ms | 2.107 s | Yes |

### Metric 5: Noise Suppression and Edge Preservation
- **Noise Suppression Definition:** High-frequency diagonal wavelet Median Absolute Deviation ($\hat{\sigma}_{\text{MAD}} = \text{median}(|W_{HH}|)/0.6745$) measured in homogeneous shadow regions (lower residual $\hat{\sigma}_{\text{MAD}}$ and higher gain-normalized suppression dB are better).
- **Edge Preservation Definition:** Edge Preservation Index (EPI $\in [0, 1]$, higher is better) combining Sobel gradient magnitude correlation and FSIM-style boundary similarity.

| Approach | Residual Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | Gain-Normalized Noise Suppression (dB) ↑ | Noise Cleanliness Score $[0,1]$ ↑ | Edge Preservation Index (EPI) ↑ |
|---|---:|---:|---:|---:|
| Classical CV Only | 10.30 | +-4.48 dB | 0.884 | 0.7084 |
| Deep Learning Only (Adapted) | 0.80 | +22.44 dB | 0.998 | 0.9338 |
| Adaptive Hybrid (ARHR Proposed) | 3.24 | +9.32 dB | 0.986 | 0.9004 |

---

---

## 3. Supplementary Smartphone Benchmark — SMALL v1.0.0 (Mendeley Data)
- **Dataset:** SMALL v1.0.0 (Smartphone Multi-Device Low-Light Dataset, 5 phone models across 371 scenes).
- **Split:** Official Test Split (50 low/reference pairs, scene-disjoint).
- **Protocol:** Evaluated strictly as an independent supplementary mobile-camera benchmark per `SUPPLEMENTARY_DATASET_RESEARCH.md`.

| Approach | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Processing Latency (ms / s) ↓ | 5a. Noise Suppression (Residual $\hat{\sigma}_{\text{MAD}}$ ↓ / Score) | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Classical CV Only** | 9.64 | 0.2128 | 1.0594 | 9.8023 | 131.6 ms (0.132 s) | 11.51 (score 0.785) | 0.2547 |
| **Deep Learning Only (Adapted)** | 11.98 | 0.5059 | 0.6883 | 5.4256 | 3796.2 ms (3.796 s) | 0.84 (score 0.996) | 0.3012 |
| **Adaptive Hybrid (ARHR Proposed)** | 11.19 | 0.3910 | 0.8470 | 7.7293 | 4206.5 ms (4.207 s) | 4.25 (score 0.954) | 0.3093 |

## 4. Ablation Study

### Ablation Study — Component Contribution Analysis

| Experiment / Configuration | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Latency (ms) ↓ | 5a. Shadow Noise $\hat{\sigma}_{\text{MAD}}$ ↓ | 5b. Edge Preservation (EPI ↑) |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0: Fixed Gamma (0.5) | 9.72 | 0.4411 | 0.3855 | 7.6848 | 8.6 | 3.89 | 0.8100 |
| E1: Fixed Classical (Non-Adaptive) | 12.07 | 0.5071 | 0.4163 | 8.1166 | 63.6 | 4.43 | 0.7942 |
| E2: Adaptive Classical | 14.74 | 0.4447 | 0.4999 | 9.0019 | 63.4 | 10.30 | 0.7084 |
| E3: Pretrained Deep Backbone | 27.67 | 0.8824 | 0.0636 | 4.8474 | 2042.0 | 1.42 | 0.9338 |
| E4: Fine-Tuned Deep Backbone | 27.52 | 0.9002 | 0.0742 | 4.8376 | 1996.0 | 0.80 | 0.9338 |
| E5: 50/50 Uniform Blend | 20.58 | 0.6643 | 0.2948 | 7.7122 | 2040.2 | 5.99 | 0.8345 |
| E6: Illumination-Guided Fusion | 26.20 | 0.8633 | 0.1007 | 5.4992 | 2107.8 | 2.29 | 0.9255 |
| E7: Guided Fusion (Illum+Noise+Edge) | 22.91 | 0.8062 | 0.1612 | 6.4326 | 2124.6 | 3.24 | 0.9004 |
| E8: Guided Fusion + Quality Guard | 22.91 | 0.8062 | 0.1612 | 6.4326 | 2121.6 | 3.24 | 0.9004 |

---

## 5. Documented Failure-Case Analysis

| Failure Case | Input Condition | Observed Behavior | Root Technical Cause |
|---|---|---|---|
| **Case 1 — Crushed Blacks / Zero-Bit Quantization Shadows** | Extremely dark regions where raw RGB intensities are quantized to (0, 0, 0). | Loss of fine micro-texture and posterization in deepest shadow pockets. | When photon counts fall below the sensor read-noise floor and quantize to 0, true spatial signal is physically absent; non-hallucinatory restoration avoids inventing synthetic textures. |
| **Case 2 — Severe Mixed-Illuminant Scenes** | Simultaneous warm sodium-vapor streetlights (2200K) and cool fluorescent/LED signage (6000K). | Residual warm or cool tint in localized transition zones. | Global and smooth spatial white-balance priors assume slowly varying illumination chromaticity and can under-correct sharp dual-illuminant boundaries. |
| **Case 3 — Low-Light Motion Blur & Defocus** | Handheld long-exposure capture with camera shake or moving subjects. | Restored brightness reveals motion-blurred edges cleanly without deblurring them. | Retinex decomposition and HVI-CIDNet restore photometric illumination and suppress sensor noise, but do not perform blind motion point-spread-function (PSF) deconvolution. |
| **Case 4 — Extreme High-ISO Sensor Noise on Fine Foliage/Fabric** | ISO > 12800 capture where chroma/luma noise amplitude matches fine texture gradients. | Trade-off between slight residual grain or mild softening of stochastic micro-textures. | When noise variance overlaps the spatial frequency band of fine stochastic textures, edge-preserving filters face an information-theoretic limit. |

---

## 6. Earlier local callback timing on synthetic inputs
The earlier 20-image audit (`tools/audit_live_latency.py`) called the app handler directly after three warmups. It measured local processing on synthetic arrays. Hosted upload and downloads are recorded separately in `outputs/submission/hosted_latency.json`.

- **Total local handler calls**: 20 / 20 successful
- **Zero Crashes**: **PASS** (0 crashes observed)
- **Zero OOM Exceptions**: **PASS** (0 memory errors observed)
- **All Inferences < 10.0 s Target**: **PASS** (100% under SLA threshold)
- **Minimum Latency**: 2122.5 ms (2.12 s)
- **Median Latency**: 2996.6 ms (3.00 s)
- **P95 Latency**: 4719.8 ms (4.72 s)
- **Maximum Latency**: 4750.6 ms (4.75 s)
- **Audit Record Artifact**: `outputs/live_latency_audit.json`

