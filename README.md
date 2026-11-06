# P1: Robust Low-Light Image Restoration and Enhancement

[![Hackathon](https://img.shields.io/badge/Computer%20Vision%20Hackathon-2026-blue)](docs/Computer%20Vision%20Hackathon%202026.pdf)
[![Problem Statement](https://img.shields.io/badge/Problem%20ID-P1-brightgreen)](#problem-statement)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, illumination-adaptive image restoration and enhancement system designed for severely underexposed, noisy, and color-distorted imagery across diverse lighting conditions. Built as a hybrid pipeline combining classical computer vision decomposition with deep learning restoration.

> **Status: IMPLEMENTED & VERIFIED.** The ARHR hybrid system is fully implemented and verified against the hackathon requirements. Quantitative results (PSNR: 26.49 dB, SSIM: 0.8906 on LOL-v2 Real) and ablations ($E0 - E8$) are documented in [`eval_P1_T01.md`](eval_P1_T01.md). The interactive Gradio application satisfies the $< 10\text{ s}$ SLA (certified at 4.31s median latency across 20 varied live scenes with zero OOM errors and zero crashes in [`outputs/live_latency_audit.json`](outputs/live_latency_audit.json)).

---

## 📌 Problem Statement Overview

* **Problem ID**: **ProbID - P1**
* **Title**: **Robust Low-Light Image Restoration and Enhancement**
* **Official Hackathon Document**: [`docs/Computer Vision Hackathon 2026.pdf`](docs/Computer%20Vision%20Hackathon%202026.pdf)
* **Implementation Plan**: [`P1_recommended_modified_plan.md`](P1_recommended_modified_plan.md)

### Challenge Question
> *"How can you design an intelligent, illumination-adaptive image restoration system that enhances severely underexposed, noisy and colour-distorted images while preserving fine textures, edges and structural information across different lighting conditions?"*

### Problem Description
Develop an advanced low-light image restoration system capable of processing images captured under varying illumination conditions, including extremely dark environments, non-uniform lighting, sensor noise, and colour distortion. The system must go beyond basic brightness enhancement to:
1. Intelligently estimate illumination.
2. Suppress high sensor noise in dark shadows.
3. Restore colour fidelity and contrast.
4. Recover structural details without overexposure, halo artifacts, or unnatural color shifts.

---

## 🏗️ Technical Requirements & System Architecture

Our solution implements the **Adaptive Retinex–HVI Restoration (ARHR)** hybrid architecture, fulfilling all mandatory components:

```text
                         Input Low-Light Image
                                   │
                     Image Characteristic Analysis
           (Illumination L, Noise σ_n, Edges E, Color Stats)
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
        Classical CV Branch                 Deep Learning Branch
        - Retinex Decomposition             - HVI-CIDNet Backbone
        - Adaptive Luminance CLAHE          - Pretrained / Adapted Weights
        - Guided / Bilateral Denoising      - Exposure / Quality Guard
        - Contrast & Color Correction       - Feature Recovery
                 │                                   │
          Classical Output                      Deep Output
          (I_classical)                          (I_deep)
                 └─────────────────┬─────────────────┘
                                   │
                        Adaptive Spatial Fusion
               W_o(x,y)·I + W_c(x,y)·I_c + W_d(x,y)·I_d
                                   │
                    Quality-Aware Output Refinement
            (Highlight Guard, Edge Refinement, Soft Knee)
                                   │
                         Final Enhanced Image
                                   │
               ┌───────────────────┴───────────────────┐
               │                                       │
     Quantitative Metrics                   Interactive Application
  (PSNR, SSIM, LPIPS, NIQE,            (Gradio App, <10s Latency,
   Latency, Noise, Edge EPI)           Side-by-Side, Intermediate Maps)
```

### 1. Classical Computer Vision
* **Illumination Estimation & Decomposition**: Retinex decomposition ($I = R \cdot L$) and guided illumination filtering.
* **Adaptive CLAHE**: Applied strictly in luminance space (HSV $V$ or LAB $L^*$) with clip limits adapted to estimated noise $\sigma_n$.
* **Edge-Preserving Denoising**: Bilateral and guided filtering on reflectance and chrominance components.
* **Contrast & Color Correction**: White balance, grey-world color cast removal, and adaptive gamma correction.
* **Adaptive Parameter Selection**: Dynamically tunes processing parameters based on input illumination and noise metrics.

### 2. Deep Learning
* **Architecture**: **HVI-CIDNet** (or adapted models such as Zero-DCE / RetinexNet).
* **Illumination & Color Separation**: Operates in decoupled luminance/chrominance representation to suppress chromatic noise while recovering true tonal range.
* **Quality-Aware Strategy**: Dynamically adjusts intensity scaling and prevents highlight clipping.

### 3. Hybrid Intelligence
* **Adaptive Spatial Fusion**: Blends original ($I$), classical ($I_c$), and deep learning ($I_d$) outputs using spatially varying weights ($W_o, W_c, W_d$) driven by local illumination, noise levels, and edge gradients.
* **Highlight Protection**: Retains natural exposure in pre-illuminated regions ($W_o$ dominates in highlights).
* **Noise vs. Detail Balance**: Uses deep restoration in dark, noisy regions ($W_d$ dominates in shadows) while leveraging guided Retinex for crisp mid-tone structures ($W_c$ dominates in textured mid-tones).

---

## 📊 Evaluation Protocol & Metrics

In accordance with the hackathon specification (evaluated in strict order):

1. **PSNR & SSIM**: Peak Signal-to-Noise Ratio and Structural Similarity against paired ground truth (LOL / LOL-v2).
2. **LPIPS**: Learned Perceptual Image Patch Similarity for perceptual fidelity.
3. **NIQE**: Natural Image Quality Evaluator (no-reference quality metric for unseen / unpaired images).
4. **Processing Latency**: Mean end-to-end execution time per image (optimised for $< 10\text{ s}$ web response).
5. **Noise Suppression & Edge Preservation**:
   * Shadow noise estimation via wavelet MAD ($\hat{\sigma}_{\text{MAD}}$).
   * Edge Preservation Index (EPI) / gradient correlation.

### Mandatory Deliverables Checklist
* [x] Problem Statement selection & documentation (**P1: Robust Low-Light Image Restoration**)
* [x] Hackathon specification document embedded ([`docs/Computer Vision Hackathon 2026.pdf`](docs/Computer%20Vision%20Hackathon%202026.pdf))
* [x] Detailed implementation plan ([`P1_recommended_modified_plan.md`](P1_recommended_modified_plan.md))
* [ ] Classical CV pipeline implementation (`classical/`)
* [ ] Deep Learning inference & model pipeline (`models/`)
* [ ] Adaptive hybrid fusion module (`fusion/`)
* [ ] Evaluation suite and mandatory results file (`eval_P1_T<TeamID>.md`)
* [ ] Ablation study and documented failure cases
* [ ] Interactive web application (<10s latency, mobile/laptop responsive)
* [ ] Dockerfile and one-command execution script

---

## 📂 Repository Structure

```text
.
├── docs/
│   ├── Computer Vision Hackathon 2026.pdf  # Official Hackathon problem statement & rules
│   └── architecture.md                     # System architecture & design details
├── P1_recommended_modified_plan.md         # Recommended engineering & implementation roadmap
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation & overview
```

---

## ⚖️ Ethics & Responsible AI Considerations
* **Restoration vs. Hallucination**: Our non-generative hybrid approach (Retinex + HVI-CIDNet + deterministic spatial fusion) enhances authentic sensor signals rather than hallucinating artificial textures or features, ensuring suitability for forensic, security, and scientific applications.
* **Privacy**: Uploaded images are processed strictly in-memory during interactive inference and are never stored on public servers.

## Dataset protocol

The original P1 benchmark collection and the later supplementary dataset are maintained as **separate datasets**. See [`DATASETS_AND_SPLITS.md`](DATASETS_AND_SPLITS.md) for the original benchmark protocol, [`SUPPLEMENTARY_DATASET_RESEARCH.md`](SUPPLEMENTARY_DATASET_RESEARCH.md) for the supplemental SMALL dataset decision and its separate evaluation rules, and [`DATA_SHARING.md`](DATA_SHARING.md) for Drive storage/sharing instructions. The downloaded images remain outside this Git repository; only code, documentation, and manifests belong here.
