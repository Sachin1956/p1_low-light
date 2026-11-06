# P1 — Recommended Modified Implementation Plan

## Robust Low-Light Image Restoration and Enhancement

**Recommended direction:** Build the Adaptive Retinex–HVI Restoration (ARHR) system from Plan 1, with a simpler, rule-based adaptive fusion stage and lightweight image-quality-aware feedback as the first working implementation.

The aim is to satisfy all mandatory P1 requirements and rubric criteria while keeping the project implementable, testable, explainable, and reliable during the live demonstration (responding within the strict $< 10\text{ s}$ rubric limit).

---

## 1. Final system architecture

The proposed system has five stages:

1. **Image analysis:** estimate illumination, noise, edges, exposure, and colour characteristics.
2. **Parallel restoration:** produce a classical-CV result and an HVI-CIDNet result.
3. **Adaptive hybrid fusion:** combine the original, classical, and deep-learning outputs using spatially varying weights.
4. **Image-quality-aware output refinement:** protect highlights, preserve colour/edges/details, and apply a lightweight quality-aware check (preventing overexposure/clipping and excessive noise amplification).
5. **Evaluation and application:** compare the three approaches, generate the mandatory `eval_P1_T<TeamID>.md` deliverable, expose intermediate visualizations, and deploy the application.

```text
                         Input Image
                              |
                    Image Characteristic
                          Analysis
             (illumination / noise / edges / colour)
                              |
                +-------------+-------------+
                |                           |
         Classical CV Branch          Deep Learning Branch
         - Retinex decomposition      - HVI-CIDNet
         - Adaptive CLAHE              - Pretrained checkpoint
         - Guided/bilateral filtering - Fine-tuning on paired data
         - Colour correction          - Quality-aware scaling
                |                           |
         Classical Output              Deep Output
                +-------------+-------------+
                              |
                     Adaptive Spatial Fusion
                  (original + classical + deep)
                              |
                Quality-Aware Output Refinement
       (highlight clipping guard / colour / edge / detail)
                              |
                    Final Enhanced Image
                              |
              Metrics / Visualizations / Web App
```

This is the recommended project architecture. Components marked as optional below should only be added after this core pipeline works.

---

## 2. Modifications to the original Plan 1

| Component | Original Plan 1 | Recommended modification | Priority |
|---|---|---|---|
| Classical CV | Retinex, CLAHE, guided/bilateral filtering, gamma and colour correction | Keep all required methods. Adapt parameters using image illumination and noise estimates. | Essential |
| Deep-learning backbone | HVI-CIDNet | Keep HVI-CIDNet as the sole primary restoration backbone. | Essential |
| FiLM conditioning | Feed classical maps into HVI-CIDNet feature stages | Defer until the unmodified pretrained model and baseline pipeline work. | Optional improvement |
| Hybrid fusion | Train a U-Net to predict three spatial weight maps | Begin with rule-based illumination-, noise-, and edge-guided spatial fusion. | Essential first version |
| Learned fusion network | Separate U-Net fusion model | Consider only if rule-based fusion is stable and there is enough time and training data. | Optional improvement |
| Additional models | Retinexformer and Zero-DCE++ comparisons | Do not integrate them into the production pipeline. Add at most one as an extra baseline if time permits. | Optional |
| Training | Fine-tune HVI-CIDNet and train fusion network | Fine-tune the restoration backbone first. Train a fusion network only as a later experiment. | Essential, staged |
| Loss function | Seven-component composite loss | Start with Charbonnier and SSIM losses. Add perceptual, colour, or gradient terms only when validation experiments justify them. | Essential, staged |
| Image-quality-aware strategy | Automatic NIQE-based corrective pass | Use a fast, deterministic quality-aware guard (exposure clipping, colour shift, and shadow noise check) in the core loop to satisfy the mandatory P1 requirement without adding slow iterative re-inference. Display NIQE as a diagnostic metric. | Essential |
| Deployment & latency | Multi-view diagnostic application | Start with Gradio, cap live inference resolution (e.g., max dimension $1024\text{--}1280\text{ px}$) so end-to-end response is strictly $< 10\text{ seconds}$ on phone and laptop. | Essential |

### Why simplify fusion first?

Training a separate fusion U-Net introduces another model, training pipeline, checkpoint, and potential failure point. A rule-based fusion mechanism lets the team demonstrate genuine hybrid intelligence earlier. The learned fusion network remains a possible upgrade, not a prerequisite for the first working submission.

---

## 3. Stage 1 — Image characteristic analysis

For an input image, calculate or estimate:

- **Illumination:** a spatial illumination map $L(x,y)$ that identifies dark, mid-tone, and already well-lit regions.
- **Noise:** a spatial noise estimate $\sigma_n(x,y)$, especially in shadow regions (e.g., local high-frequency residual or wavelet-based Median Absolute Deviation estimator).
- **Edges and details:** gradient magnitude $E(x,y)$ (Sobel/Scharr) to identify important boundaries and textures.
- **Exposure statistics:** mean and percentile luminance, underexposed-pixel proportion ($L < 0.15$), and clipped-highlight proportion ($L > 0.95$).
- **Colour characteristics:** channel ratios, saturation, and grey-world colour-cast indicators.

These measurements are used to adapt classical parameters and guide fusion. They should also be exposed in the application as intermediate visualizations where useful.

---

## 4. Stage 2 — Parallel restoration branches

### 4.1 Classical computer-vision branch

Implement the mandatory classical components:

1. Illumination estimation and Single/Multi-Scale Retinex-based decomposition ($I = R \cdot L$).
2. Guided filtering (or fast bilateral filtering) to refine illumination $L$ while respecting edges and preventing halo artifacts.
3. Adaptive CLAHE applied to the luminance channel (HSV $V$ or LAB $L^*$) rather than independently to RGB channels, with clip limit adapted to the estimated noise level.
4. Bilateral or guided filtering on the reflectance/chrominance components for edge-preserving denoising.
5. Adaptive gamma correction ($\gamma$ dynamically chosen from mean/median illumination).
6. Contrast and colour correction (white-balance / grey-world / saturation restoration).

Use estimated image characteristics to adapt parameters. For example, lower the CLAHE clip limit and increase denoising strength when the input has high shadow noise $\sigma_n$, so noise is not amplified.

Save the result as `I_classical` so it can be compared independently with the deep-learning and hybrid outputs.

### 4.2 Deep-learning branch

Use **HVI-CIDNet** as the primary deep-learning model.

Recommended order:

1. Set up the official implementation and its compatible environment.
2. Load and run the released checkpoint.
3. Verify inference on sample images and implement an **image-quality-aware intensity scaling / exposure control** parameter (adjusting enhancement strength based on input illumination and output clipping check, fulfilling the mandatory *"image-quality-aware enhancement strategy"* requirement).
4. Evaluate the checkpoint on the chosen validation split.
5. Fine-tune on the training portions of LOL-v2 Real and LOL-v2 Synthetic if resources and time permit.
6. Evaluate again using exactly the same preprocessing and metric protocol.

Keep the model's standalone result as `I_deep`.

**Important:** Do not modify the internal model architecture before establishing a reproducible baseline. This gives the team a working deep-learning branch even if later experiments are unsuccessful.

---

## 5. Stage 3 — Adaptive hybrid fusion

The first version should use spatially varying, rule-based weights for three candidate images:

- `I`: original input image.
- `I_classical`: classical-CV output.
- `I_deep`: HVI-CIDNet output.

The final image is:

$$
I_{\text{hybrid}}(x,y) = W_o(x,y)I(x,y) + W_c(x,y)I_{\text{classical}}(x,y) + W_d(x,y)I_{\text{deep}}(x,y)
$$

with:

$$
W_o(x,y) + W_c(x,y) + W_d(x,y) = 1
$$

The weights depend on the estimated illumination $L(x,y)$, noise $\sigma_n(x,y)$, edge confidence $E(x,y)$, and highlight mask.

### Intended behaviour

- **Already well-lit regions ($L(x,y)$ high):** assign higher $W_o(x,y)$ to retain more of the original input, avoiding overexposure and clipped highlights.
- **Very dark or noisy shadow regions ($L(x,y)$ low, $\sigma_n(x,y)$ high):** assign higher $W_d(x,y)$ to rely more on HVI-CIDNet where it suppresses colour noise and recovers shadow features better than CLAHE/Retinex.
- **Mid-tone and high-gradient structural regions ($E(x,y)$ high, moderate $L$):** assign meaningful $W_c(x,y)$ where classical guided-filter Retinex preserves crisp local contrast and natural edge gradients.
- **Smooth weight transitions:** apply a fast guided filter or Gaussian smoothing to the raw weight maps $W_o, W_c, W_d$ before blending so no seam or halo artifacts appear.

Tune the fusion rules on the validation set and verify them with visual inspection and metrics.

### Optional later upgrade: learned fusion

If the rule-based version works reliably, train a small U-Net-style fusion model to predict the three spatial weight maps. Compare it against the rule-based fusion using the same validation protocol. Keep the learned version only if it produces a measurable improvement without harming latency or reliability.

---

## 6. Stage 4 — Image-quality-aware output refinement

Apply conservative, quality-aware final corrections:

- **Quality-aware highlight and colour guard:** check the fused output's highlight clipping ratio ($>0.98$ luminance) and channel saturation shift. If clipping exceeds a threshold (e.g., $2\%$), automatically blend back unclipped luminance from $I$ or apply soft knee tone-mapping.
- **Edge-aware chroma/luma smoothing:** apply light bilateral/guided smoothing only in flat, high-noise regions where residual noise remains above threshold.
- **Detail preservation:** apply mild unsharp masking only where edge confidence $E(x,y)$ is high and noise $\sigma_n(x,y)$ is low.
- **Honest physical boundaries:** do not claim that lost information can always be recovered. Extremely crushed blacks ($0$ intensity across RGB), severe motion blur, and extreme sensor noise should be documented as limitations or failure cases.

---

## 7. Dataset and training strategy

| Dataset | Recommended role |
|---|---|
| LOL-v2 Real | Primary paired training/fine-tuning and evaluation resource, with official test data kept untouched. |
| LOL-v2 Synthetic | Additional training diversity and a separate evaluation benchmark. |
| LOL-v1 | Additional benchmark comparison (official link in Hackathon Datasheet). |
| MIT-Adobe FiveK | Optional supplementary tonal/colour experiment (listed in Hackathon Datasheet); do not treat expert-retouched images as equivalent to paired low-light ground truth. |
| MILL / Unpaired test images | Optional independent generalization evaluation on unseen low-light images. |

### Data-splitting rules

- Create the validation split only from the official training data.
- Keep official test sets untouched until final evaluation.
- Do not use test images to select parameters, choose checkpoints, or fine-tune.
- Report which dataset and split produced every metric.
- Do not calculate full-reference metrics such as PSNR, SSIM, or LPIPS on unpaired images without a genuine corresponding reference (use NIQE, noise suppression, edge preservation, and latency for unpaired images).

### Training order

1. Establish the classical baseline (`I_classical`).
2. Reproduce HVI-CIDNet inference with its released checkpoint (`I_deep`).
3. Record both standalone baselines on the validation split.
4. Implement rule-based adaptive fusion (`I_hybrid`) and verify that it improves over or balances the standalone branches.
5. Fine-tune HVI-CIDNet on the training split if time and GPU resources permit.
6. Only then attempt FiLM conditioning or learned fusion.

### Loss functions

Start with:

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{\text{Charbonnier}} + \lambda_2 (1 - \text{SSIM})
$$

Consider adding LPIPS/perceptual, colour-angle, or gradient losses one at a time. Retain an additional loss only when the validation results show a useful contribution.

---

## 8. Evaluation plan & mandatory deliverable format

### 8.1 Approaches to compare

| Approach | System to evaluate |
|---|---|
| Classical only | Adaptive Retinex + guided filter + luminance CLAHE + colour correction (`I_classical`) |
| Deep learning only | HVI-CIDNet (`I_deep`) |
| Hybrid | Classical branch + HVI-CIDNet + adaptive spatial fusion + quality-aware refinement (`I_hybrid`) |

### 8.2 Exact metric order and concrete definitions

To satisfy the P1 specification (Page 2) and the Instruction Sheet rubric (Page 15, worth 25 points), compute and report all metrics in the **exact order listed in the problem statement**:

| Order | Metric | Concrete definition / implementation | Direction |
|---|---|---|---|
| 1 | **PSNR & SSIM** | Peak Signal-to-Noise Ratio (dB) and Structural Similarity Index on RGB against paired ground truth (`skimage.metrics` / `piq`). | Higher is better |
| 2 | **LPIPS** | Learned Perceptual Image Patch Similarity (AlexNet or VGG backbone via `lpips` library) against paired ground truth. | Lower is better |
| 3 | **NIQE** | Natural Image Quality Evaluator (`pyiqa` / `piq` / `skimage`) computed on restored output (works on both paired and unpaired unseen inputs). | Lower is better |
| 4 | **Processing latency** | Mean end-to-end execution time per image (ms or s), explicitly stating input resolution and hardware (CPU vs. GPU). | Lower is better |
| 5 | **Noise suppression & Edge preservation** | **Noise suppression:** Reduction ratio or post-restoration shadow noise std $\hat{\sigma}_{\text{MAD}}$ estimated via high-frequency wavelet Median Absolute Deviation in flat regions.<br>**Edge preservation:** Edge Preservation Index (EPI) — correlation of Sobel/Laplacian gradients between structure-normalized input/reference and restored output (or Feature Similarity Index, FSIM). | Lower residual noise $\hat{\sigma}$ / Higher suppression;<br>Higher EPI (closer to $1.0$) |

### 8.3 Mandatory `eval_P1_T<TeamID>.md` file format

Per Page 15 of the Hackathon PDF, the repository **must** include an evaluation file named `eval_P1_T<TeamID>.md` (e.g., `eval_P1_T01.md`):

- **Line 1 MUST be:** All team members' USNs separated by commas (e.g., `1RV23CS001, 1RV23CS002, 1RV23CS003, 1RV23CS004`).
- **Subsequent sections:** Comparative evaluation tables for Classical-only, Deep-Learning-only, and Hybrid approaches with metrics in the exact order shown in Section 8.2, along with dataset split and hardware specifications.

### 8.4 Ablation study

Use a progressive experiment table such as:

| Experiment | Configuration | Purpose |
|---|---|---|
| A | Classical CV only (fixed parameters) | Baseline without adaptive parameter selection. |
| B | Classical CV only (adaptive parameters) | Measure contribution of illumination/noise-adaptive parameters. |
| C | HVI-CIDNet only | Establish the deep-learning baseline. |
| D | Classical + Deep, uniform $50/50$ blend | Show the behaviour of non-adaptive fusion. |
| E | Illumination-guided spatial fusion | Measure the contribution of illumination awareness. |
| F | Full ARHR Hybrid (Illumination + noise + edge-guided fusion + quality-aware refinement) | Measure the complete system's contribution. |

Only include experiments that have actually been implemented and evaluated. Do not pre-fill the table with expected improvements.

### 8.5 Failure-case analysis

Prepare visual examples and explain at least **three real failure cases**:

1. **Crushed blacks / clipped shadows:** regions where raw pixel values are near $(0,0,0)$ and quantization has permanently destroyed signal.
2. **Mixed lighting with multiple illuminants:** scenes combining warm tungsten/sodium streetlights with cool daylight/fluorescent sources, causing localized colour casts.
3. **Extreme sensor noise vs. fine texture:** dark foliage, fabric, or hair where heavy denoising softens fine structure or mild denoising leaves chroma speckles.
4. **Motion blur or defocus in low light:** long-exposure camera shake or subject motion, which illumination restoration makes more visible rather than deblurring.
5. **Strong point light sources in dark scenes:** streetlights or headlights that risk localized blooming or halo transitions around saturated regions.

For each case, show the input, output, zoomed region, observed artifact, and root technical cause.

---

## 9. Application, deployment, and the $< 10\text{ s}$ rubric rule

Start with **Gradio** to reduce frontend development time and ensure out-of-the-box mobile and desktop responsiveness.

### Minimum useful interface

- Upload a JPG, PNG, or WebP image (plus quick-select preset test images for instant judge testing).
- Display the original and final enhanced image side-by-side with an interactive before/after slider.
- Show separate **Classical**, **Deep Learning (HVI-CIDNet)**, and **Hybrid** outputs for direct visual comparison.
- Display intermediate visualizations: **Illumination Map**, **Noise Map / Edge Map**, and **Spatial Fusion Weights ($W_o, W_c, W_d$)**.
- Display real-time calculated diagnostics: **Processing Latency (ms)**, **NIQE**, **Noise Estimate**, **Edge Preservation**, and exposure stats.
- Provide a download button for the restored image.
- Include clear loading indicators and user-friendly error messages for invalid/corrupted files.

### Strict rubric reliability & latency requirements (15 Points)

- **$< 10\text{ seconds}$ response budget:** The rubric explicitly requires that the *"public URL works on first click from phone and laptop, responds within about 10 seconds."*
  - Automatically cap or downscale oversized uploads (e.g., if $\max(H, W) > 1024\text{ px}$ in live demo mode, resize proportionally for inference or compute classical/weight maps at half resolution and upsample via guided filtering).
  - Keep fast no-reference diagnostics enabled by default, and make any slow multi-scale CPU metric optional if running on a CPU-only cloud instance.
- **Startup preloading:** Load HVI-CIDNet weights once at application startup (`@lru_cache` / global singleton), never per request.
- **Robust input handling:** Gracefully catch non-image uploads, grayscale/RGBA images (convert RGBA $\to$ RGB, 16-bit $\to$ 8-bit normalized), tiny images, and EXIF orientation tags.
- **Reproducible one-command setup:** Provide both a `Dockerfile` and a single startup command (e.g., `python app.py` or `bash run.sh`) in the repository root.
- **Security:** Never commit API keys, tokens, or secrets to the repository.

---

## 10. Project documentation & ethics (15 Points)

To hit the top band of the **Project Documentation** rubric (Page 15–16), the repository must include:

1. **System Architecture Diagram:** Must match the actual implemented code and pipeline stages.
2. **Design Document:** Covering problem formulation, classical CV math, HVI-CIDNet architecture, spatial fusion formulation, adaptive parameter logic, dataset split protocol, ablation study, and failure cases.
3. **Reproducible `README.md`:** Step-by-step installation, dataset preparation, evaluation command (`python evaluate.py`), and one-command app launch.
4. **Honest Limitations and Ethical Considerations:** (Explicitly checked in the rubric!)
   - *Technical limitations:* Quantization floor in zero-bit shadows, motion blur amplification, extreme mixed-illuminant white balance ambiguity.
   - *Ethical considerations:* Restoration vs. hallucination in forensic, medical, dashcam, or surveillance imagery. Emphasize why our **non-generative hybrid approach (Classical Retinex + HVI-CIDNet + spatial fusion)** is ethically safer than GAN-based enhancers (like EnlightenGAN), because it amplifies true underlying sensor signal rather than hallucinating nonexistent faces, license plates, or structures in dark regions. Also note privacy considerations (uploaded images are processed in-memory and never stored on the server).

---

## 11. Three-minute demonstration and viva plan (25 Points)

Demo and viva carry the highest weight (**25 points**) and serve as the **primary tie-breaker**.

| Time | Demonstration |
|---|---|
| 0–25 seconds | Introduce P1, the core challenge (enhancing darkness without amplifying noise or overexposing highlights), and the 5-stage ARHR hybrid architecture. |
| 25–65 seconds | Upload an **unseen low-light image** on the live public URL; show $< 10\text{ s}$ execution and the interactive before/after restoration. |
| 65–105 seconds | Walk through the intermediate visualizations: Illumination map $L$, Noise/Edge maps, and how the Spatial Fusion Weights ($W_o, W_c, W_d$) adapt across dark shadows vs. bright regions. |
| 105–145 seconds | Compare **Classical-only**, **HVI-CIDNet-only**, and **Hybrid** outputs side-by-side, referencing the quantitative benchmark table (`eval_P1_T<TeamID>.md`) and ablation results. |
| 145–180 seconds | Show a genuine **failure case** (e.g., crushed blacks or motion blur), explain its physical/mathematical cause, and state the ethical advantage of non-hallucinatory restoration. |

**Viva readiness:** Every team member must be able to explain:
- Their specific code module and contribution.
- Why CLAHE is applied in luminance space and how guided filtering prevents Retinex halos.
- How HVI-CIDNet decouples Horizontal/Vertical chrominance from Intensity.
- How each evaluation metric (PSNR, SSIM, LPIPS, NIQE, Latency, Noise Suppression, Edge Preservation) works.
- At least one failure case and why it occurs.

---

## 12. Recommended implementation checklist

### Essential — complete these first

- [ ] **Classical CV pipeline:** Retinex decomposition, guided filtering, luminance CLAHE, bilateral/guided denoising, gamma & colour correction, with adaptive parameter selection.
- [ ] **Deep Learning pipeline:** HVI-CIDNet inference using released checkpoint + lightweight quality-aware exposure/intensity control.
- [ ] **Adaptive Hybrid Fusion:** Spatial weight computation ($W_o, W_c, W_d$) guided by illumination, noise, and edge maps + highlight clipping guard.
- [ ] **Evaluation suite:** Reproducible script computing PSNR, SSIM, LPIPS, NIQE, Processing Latency, Noise Suppression ($\hat{\sigma}_{\text{MAD}}$), and Edge Preservation (EPI) in exact PDF order.
- [ ] **Mandatory Deliverable File:** `eval_P1_T<TeamID>.md` with comma-separated USNs on Line 1 and complete comparative tables.
- [ ] **Ablation study & Failure cases:** Quantitative ablation table (Experiments A–F) and visual analysis of $\ge 3$ failure cases.
- [ ] **Gradio Web App:** Side-by-side comparison, intermediate map visualizations, bad-input handling, and resolution guard for $< 10\text{ s}$ live latency.
- [ ] **Deployment & Repo:** Public hosted URL (tested on mobile + laptop), `Dockerfile` + one-command runner, architecture diagram, design doc (with Limitations & Ethics section), and `README.md`.
- [ ] **Three-minute demo rehearsal:** Timed run-through with all team members ready for viva questions.

### Differentiators — add after the core works

- [ ] Fine-tune HVI-CIDNet on LOL-v2 Real / Synthetic training split (with strict train/val split).
- [ ] Add FP16 / ONNX / `torch.compile` optimization to further cut inference latency and document the speedup.
- [ ] Test a learned spatial fusion network against rule-based fusion.
- [ ] Add FiLM conditioning using classical illumination/noise information.

### Optional — only if time permits

- [ ] Add one additional deep-learning baseline such as Zero-DCE++ or Retinexformer.
- [ ] Add multi-scale Laplacian-pyramid blending for the spatial weight maps.

---

## 13. Final recommendation

Commit to the five-stage ARHR architecture, and implement it incrementally. The first complete version should contain the adaptive classical branch, HVI-CIDNet with quality-aware scaling, and rule-based spatial fusion. Keep the live web inference under 10 seconds, format `eval_P1_T<TeamID>.md` strictly according to the instruction sheet, and document both technical failure cases and ethical considerations honestly.
