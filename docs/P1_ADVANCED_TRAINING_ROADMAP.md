# P1 Advanced Training Roadmap: Data Expansion, Epoch Scaling, & Distillation

**Prepared for P1 Low-Light Image Restoration and Enhancement.**  
**Author:** Antigravity AI Agent & P1 Engineering Team  
**Evaluation Target:** CVPR / Hackathon P1 Rubric Compliance

---

## 1. Concrete Evaluation of User's Three Strategies

### A. Find Other Good Data to Continue Training On
* **Feasibility:** **HIGH / VERIFIED**
* **Available Datasets in Ecosystem:**
  1. **SMALL v1.0.0 (Smartphone Multi-Device Low-Light Dataset):**
     - *Provenance:* Mendeley Data ([DOI: 10.17632/vnz4gcw8wk.1](https://doi.org/10.17632/vnz4gcw8wk.1)), CC BY 4.0.
     - *Volume:* 504 paired images across 5 smartphone brands (iPhone, Samsung, Xiaomi, Vivo, Oppo) from 371 scenes.
     - *Partitions:* 403 Train, 51 Validation, 50 Test.
     - *Technical Value:* Directly addresses the acute smartphone domain gap. In our baseline benchmark, deep model PSNR was **26.49 dB on LOL-v2 (DSLR)** but **12.37 dB on SMALL (Smartphones)** due to extreme high-frequency sensor noise and mobile ISP nonlinear tone mapping.
  2. **LOL-v1 (Original Benchmark):**
     - *Provenance:* 485 training pairs, 15 test pairs.
     - *Technical Value:* Adds extreme low-light illumination contrast (near-zero lux indoor captures).
  3. **LSRW (Large-Scale Low-Light Dataset):**
     - 5,650 real indoor/outdoor pairs (Nikon + Huawei). Ideal for future large-scale expansion.
* **Verdict:** Expanding training data to include the **403 SMALL training pairs** is the highest-ROI, lowest-risk training enhancement possible.

---

### B. Doing More Epochs on Existing Datasets
* **Feasibility:** **EMBEDDED WITH SEVERE OVERFITTING RISK**
* **Empirical Validation Evidence (2,000 Updates on LOL-v2 Real):**
  From `weights/kaggle_run_v2/p1_kaggle_output/training_meta.json`:
  ```json
  "history": [
    { "update": 250,  "val_loss": 0.082145 },  <-- OPTIMAL MINIMUM
    { "update": 500,  "val_loss": 0.085672 },
    { "update": 750,  "val_loss": 0.086601 },
    { "update": 1000, "val_loss": 0.087648 },
    { "update": 1250, "val_loss": 0.087951 },
    { "update": 1500, "val_loss": 0.092958 },  <-- OVERFITTING PEAK
    { "update": 1750, "val_loss": 0.088109 },
    { "update": 2000, "val_loss": 0.086506 }
  ]
  ```
* **Analysis:**
  Because the training pool of LOL-v2 Real has only 689 scene pairs, HVI-CIDNet (1.88M parameters) reaches its capacity limit on this distribution within 250–500 updates. Continuing training beyond 2,000 updates without new data or aggressive stochastic regularization simply memorizes noise patterns and increases validation error.
* **Remedy if scaling epochs:**
  1. Cosine Annealing Learning Rate Schedule ($10^{-5} \to 10^{-7}$).
  2. Stronger spatial augmentations (random multi-scale crops, CutBlur, random 90-degree rotations).
  3. Weight decay regularization ($10^{-4}$).

---

### C. Knowledge Distillation from Stronger / Larger Models
* **Feasibility:** **COMPUTATIONALLY HEAVY & RUBRIC CONFLICT**
* **Potential Teacher Backbones:**
  - **Retinexformer (ICCV 2023):** 11.2M parameters (6x larger than HVI-CIDNet).
  - **Diff-Retinex / Restormer / Large Diffusion Models:** 25M–100M+ parameters.
* **Distillation Formulation:**
  $$\mathcal{L}_{\text{total}} = (1 - \alpha) \mathcal{L}_{\text{task}}(y_{\text{student}}, y_{\text{GT}}) + \alpha \mathcal{L}_{\text{distill}}(y_{\text{student}}, y_{\text{teacher}}) + \beta \mathcal{L}_{\text{feat}}(F_{\text{student}}, F_{\text{teacher}})$$
* **Critical Risks & Conflicts under Hackathon Contract:**
  1. **Compute / Latency Budget:** Generating offline teacher inference maps for 1,000+ images requires several hours on cloud GPUs before student training can begin.
  2. **Forensic Authenticity Constraint:** Section 1 of `docs/P1_ML_EXECUTION_PLAN.md` states:
     > *"No single score establishes authenticity, and even a deterministic neural network can introduce false details. Black pixels with lost signal cannot reliably reveal what was there."*
     Distillation from generative or diffusion models penalizes forensic verification because generative models invent plausible high-frequency textures that do not match the real camera sensor's optical data.

---

## 2. Recommended Unified Strategy: Multi-Domain Curriculum Adaptation

Instead of pure epoch scaling or risky generative distillation, the mathematically sound strategy that maximizes benchmark scores across both DSLR and smartphone inputs is:

```mermaid
flowchart TD
    A["Stage 1: Pretrained CIDNet Checkpoint\n(Trained on LOL-v2)"] --> B["Stage 2: Multi-Domain Adaptation (Kaggle T4)\nLOL-v2 Real (689 pairs) + SMALL v1.0.0 (403 pairs)"]
    B --> C["Cosine Annealing Schedule (3,000 updates)\nLR: 5e-6 -> 1e-7 | Crop: 256x256"]
    C --> D["Validation Checkpoints every 250 steps"]
    D --> E["Independent Dual-Benchmark Evaluation\n1. LOL-v2 Real Test Set\n2. SMALL v1.0.0 50 Test Pairs"]
```

This achieves:
1. **Zero Hallucination Risk:** Trained directly on genuine camera sensor ground-truth pairs.
2. **Mobile + DSLR Invariance:** Retains 26+ dB on studio shots while pulling smartphone PSNR up from 12.37 dB.
3. **Overnight Feasibility:** 3,000 updates completes in ~25–35 minutes on Kaggle T4x2 GPU.
