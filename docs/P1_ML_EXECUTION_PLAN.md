# P1 ML execution plan: 18-hour submission

Prepared 2026-11-06. **18 elapsed hours from implementation start (H0) to submission (H18).** Team split: project owner handles Part A on the local RTX 3050 (4 GB) and owns Part C integration; teammate handles Part B training on Kaggle and takes over app/deployment work after the H+4 checkpoint handoff. Status: **PLAN ONLY; no implementation, training, or benchmark result is certified by this document.**

This is the execution plan for the remaining hackathon. It supersedes conflicting implementation recommendations in `P1_recommended_modified_plan.md`; the official PDF remains authoritative. Read `P1_AGENT_CONTRACT.md` before implementation or evaluation.

**Goal:** deliver a live P1 application that improves low-light images across varied illumination, suppresses noise, preserves colour and structure, and demonstrates classical-only, deep-only and adaptive hybrid results on unseen inputs. Submission is done when the release gates in section 8 and the official deliverables in section 2 have measured evidence. Part A is classical CV, including adaptive parameter choice and validation, with no neural training requirement. The Kaggle teammate owns Part B model adaptation and the deep-only checkpoint. The project owner is the Part C integrator and evaluation owner. After exporting the H+4 checkpoint, the teammate owns the app shell, host setup, deployment, and phone/laptop checks while training may continue within its cap. If these duties conflict, prioritize a working deployed service over extra training updates. The owner of each branch must deliver the shared interface below.

**Shared interface, freeze at H+1:** `enhance(rgb_uint8: HxWx3, config) -> EnhancementResult`. The result has `image` (HxWx3 sRGB uint8), `maps` (named HxW float32 arrays derived from input; Part A supplies illumination/noise/edge, Part B may supply none), `params` (actual selected settings), `elapsed_ms`, `branch_version`, `config_hash`, and `checkpoint_hash` (null for Part A). Decode and apply EXIF orientation once before this call. Output dimensions and channel order match the oriented input; there is no implicit gamma change at the boundary. Batch inference reads a manifest of image IDs and input paths and writes one lossless PNG per ID and a JSONL record with ID, input SHA256, output SHA256, branch version, config hash, checkpoint hash, milliseconds and error. Both owners run five identical handoff images (dark indoor, streetlight, colour cast, textured shadow, already bright region). No references travel with the inference handoff. The integrator tests same-size, same-channel outputs and matching IDs before fusion.

**Whole-image policy:** pad to the model's required size and crop the padding; try full-frame inference at the declared live resolution by H+2. If it exceeds memory, freeze a deterministic tile size and overlap on validation (initial trial: 256 px tiles, 32 px overlap, reflection padding and smooth overlap blending). Reconstruct exactly the input dimensions, inspect seams on a test pattern, and apply the same policy to every image in a given comparison. Record any tile-policy change as a config change and rerun comparisons. Seed stochastic libraries, use `eval()` and inference mode, and verify two runs of the same image have identical output bytes on the same machine/settings; if a backend is nondeterministic, record a measured tolerance and reason before acceptance.

## 1. Understand the problem

Low illumination weakens scene signal relative to sensor noise. Brightening increases visible noise; aggressive smoothing removes textures; global enhancement can destroy lights already near saturation. The task is spatial restoration of brightness, colour, noise and structure, not simply increasing the average pixel value.

Retinex approximates an image as reflectance times illumination. It is a useful processing model, not a unique recovery of the true scene. A classical branch estimates illumination, denoises, applies luminance CLAHE, and adjusts contrast/colour. A learned branch uses patterns learned from paired examples. HVI-CIDNet separates intensity and chromatic representations. An adaptive hybrid uses input-derived maps to combine these candidates.

Training updates weights. Validation selects weights/settings. A sealed final test estimates performance after those decisions. A downloaded checkpoint may already have seen a proposed validation image: excluding it only from our fine-tuning does not undo that exposure.

PSNR measures pixel error in dB; SSIM measures structural similarity; LPIPS measures learned perceptual differences. All three need corresponding references. NIQE is reference-free but can reward undesirable appearances. Noise estimates confuse texture with noise. No single score establishes authenticity, and even a deterministic neural network can introduce false details. Black pixels with lost signal cannot reliably reveal what was there.

### Current repository state

At planning time the repository has a README, architecture document, prior plan, licence and official PDF. It has no runnable restoration implementation, trained checkpoint or measured evaluation. Descriptions of privacy, speed and quality in those documents are intended properties, not verified achievements.

## 2. Verified requirements and corrections

The official PDF, pp. 1-2, requires classical illumination/decomposition, Retinex, CLAHE, bilateral or guided denoising, contrast/colour correction and image-adaptive settings. It also requires a suitable deep model, fine-tuning/adaptation where data is available, generalization, quality awareness, adaptive hybrid integration, comparison of classical/deep/hybrid, ablations, unseen inputs and failure analysis.

PDF pp. 15-16 require a hosted URL, source, matching architecture/design documentation, comparative evaluation, ablations and failure cases. The evaluation Markdown first line contains actual comma-separated team USNs. Use `eval_P1_T<actual-team-id>.md`, replacing the placeholder only with team-provided identity. Metrics appear in the P1 order: PSNR/SSIM, LPIPS, NIQE or justified alternative, latency, noise/edges.

The PDF says response within **about 10 seconds**, not a mathematically specified strict latency benchmark. Our engineering gate is stricter: every request in the declared live acceptance set finishes within 10 s. Its rubric prose and table disagree on point totals; do not invent a scoring formula or guarantee top-10% placement.

The prior plan incorrectly suggests that common image libraries all provide NIQE. Verify the selected implementation and its model/statistics files. Architecture descriptions must call HVI horizontal/vertical chromatic coordinates plus intensity, rather than loosely equating its letters to HSV. Retinex and guided filtering reduce some artifacts; they cannot guarantee their absence.

## 3. What “state of the art” means here

Use current research to choose a competitive, reproducible method; reserve benchmark-leading claims for comparable measured evidence. HVI-CIDNet (CVPR 2025) is our primary practical backbone, not a declaration of the November 2026 global best.

The 2026 challenge literature contains stronger task-specific systems and efficient variants. Those results use different data, hardware and protocols. Our inference is that adapting a released backbone is more feasible in 18 hours than reproducing a challenge winner. A candidate without accessible compatible weights cannot displace the working system.

| Candidate | Decision for this submission | Evidence needed to change it |
|---|---|---|
| HVI-CIDNet | Primary; official implementation and released compatible checkpoint | Valid checkpoint provenance, input-only inference, full validation metrics and speed |
| Retinexformer / a reproducible newer efficient model | At most one challenger if primary is already working by H+4 | Official code, usable licence/weights, audited training provenance and improvement under identical protocol within a 45-minute setup cap |
| 2026 challenge winner architectures | Research context; no scratch reproduction | A ready checkpoint and measured deployment advantage, not a paper rank alone |
| Diffusion/flow large-model pipeline | Outside this 18-hour implementation budget | No dependency on this for completing P1 |

Primary sources: [HVI code and evaluation options](https://github.com/Fediory/HVI-CIDNet), [HVI paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html), [NTIRE 2026 LLIE report](https://arxiv.org/abs/2604.17669), [NTIRE 2026 efficient LLIE report](https://arxiv.org/abs/2605.02212). See `p1_ml_research.md` for research notes. Published scores must never be copied into our measured results.

## 4. Build the smallest complete research system

Input RGB -> analysis maps -> classical and deep branches -> normalized spatial fusion -> quality guard -> RGB output plus real diagnostics. All three evaluated approaches share decoding, geometry and output conventions.

### Classical branch

Use float RGB in [0,1], with explicit sRGB conventions. Estimate illumination from max-channel intensity and refine with a guided filter; stabilize division with epsilon. Apply noise-aware bilateral/guided filtering, bounded gamma, luminance CLAHE, and conservative colour/contrast correction. Record actual selected parameters. Do not stack aggressive operators merely to tick requirements. Preserve the intermediate reflectance/illumination result for explanation.

Estimate local high-frequency noise on flat patches, with a validity/confidence map. Sobel magnitude alone is not a reliable texture confidence in noisy shadows. Treat maps as proxies; mark invalid estimates rather than inventing values. Use bounded, validation-tuned mappings from luminance/noise to gamma, denoising strength and CLAHE clip limit. Global grey-world correction is optional within the colour-correction module and must be bounded because mixed lighting violates its assumptions.

**Done:** every required classical operation is exercised by the real path; finite same-shape outputs; maps align with input; at least two controlled inputs produce different recorded parameters; the fixed-parameter ablation and adaptive version both have saved predictions.

### Deep branch and resource budget

Pin official source revision, checkpoint URL/hash, architecture and training dataset provenance. Explicitly reject HVI's disclosed `LOLv1/test_finetuning.pth` for independent LOL-v1 evaluation: its README says it was trained on that test set. Audit all other checkpoint histories rather than assuming cleanliness. Reproduce preprocessing and loading before changing it. Strictly audit brightness corrections: the HVI repository offers GT-mean adjustment. Disable all ground-truth-dependent output adjustment. Ground truth enters only training loss or a separate metric process.

Run `eval()` and inference mode. Pad to the model's required multiple and crop padding away; never resize to match a reference. Begin FP32, then compare AMP on the same inputs. Do not assume all HVI conversion operations are stable in FP16. Keep sensitive colour operations in FP32 if needed. Missing/incompatible weights are an explicit error.

The Kaggle teammate measures the allocated accelerator first and uses its actual VRAM, quota and time limit for batch/crop sizing. Start conservatively at batch 1, 128x128 synchronized paired crops, effective batch 8 through accumulation, then increase only after measuring peak allocated/reserved memory and step time. On OOM reduce crop size and record the change. Local 4 GB training is a fallback after Kaggle access failure, not a parallel requirement. Whole-image inference may exceed crop training memory even when training fits.

Kaggle is Part B's primary training environment: within 30 minutes the teammate verifies account/GPU eligibility, a usable session, GPU type/VRAM, remaining quota, Internet access where permitted for weights, dataset access and checkpoint export. Otherwise Part B reports the blocker and switches to the local 4 GB fallback if available. The teammate exports portable weights, config, hashes, logs and the exact data manifest. Part B is not accepted until the exported checkpoint runs through the shared interface outside Kaggle. A notebook screenshot or Kaggle-only output cannot complete the handoff.

**Done:** a real loaded checkpoint restores at least five varied images; loading checks pass; no reference path is supplied to inference; preprocessing is documented; peak memory and wall time are measured at the proposed live resolution; no silent fallback occurs.

### Adaptation/fine-tuning

Attempt real adaptation when permitted paired data is accessible. Use one dataset domain first (prefer LOL-v2 Real with audited split/checkpoint provenance); do not combine Real and Synthetic by default. Training on synthetic data first and then real data is a later controlled experiment, not a requirement for this deadline.

Start with the official compatible training recipe where feasible. A simplified fine-tune is a separately labelled experiment: Adam, learning rate 1e-5, weight decay 0, clip gradient norm to 1.0, Charbonnier loss `mean(sqrt((pred-target)^2+1e-6))`. Keep this plain initial loss; test `+0.1*(1-SSIM)` only if a complete validation comparison fits. Additional perceptual/gradient/colour losses need isolated evidence. Never claim this is the paper reproduction recipe.

Use synchronized crops/flips/rotations for both images. Avoid independent colour jitter or synthetic darkening in gamma-encoded space labelled as a physically accurate sensor simulation. Seed Python/NumPy/Torch, record nondeterministic operations, save optimizer/scheduler/RNG state for resume.

Run a 100-step throughput/memory pilot. Validate every 250 optimizer updates, target at most 2,000 updates initially, and cap training at four elapsed hours or H+10, whichever arrives first. This is a short adaptation of a pretrained model, not training a new model or a full paper reproduction; small or zero gains are plausible. The Kaggle teammate exports a first usable checkpoint and five shared-interface predictions by H+4, even if training continues. Save best/last checkpoint and actual completed steps. Retain the pretrained checkpoint as comparator. A plateau can stop a run after three validations without improvement. Add EMA only if its extra memory fits and compare the EMA model explicitly.

**Done:** nonzero gradients and actual parameter changes are verified; a tiny training batch can reduce loss by at least 50% within 200 updates in a dedicated debug run; debug weights are discarded; full-run logs, before/after checkpoint hashes and complete validation predictions exist. Failure to beat pretrained weights is a valid result, not a reason to conceal adaptation. No-data or no-compute means BLOCKED for training, with the reason recorded; model integration alone does not certify training.

### Adaptive fusion and quality awareness

For candidates original O, classical C and deep D, calculate three smooth logits from input illumination, noise confidence and structural confidence. Smooth logits with input-guided filtering and apply a softmax over candidates so weights are nonnegative and sum to one. Fuse corresponding RGB values using one scalar weight per pixel per candidate. Define and freeze the processing colour space; begin sRGB to match checkpoint outputs, and do not silently switch blend space between experiments.

Prefer original in already exposed regions, deep in noisy darkness and classical where validation supports local contrast recovery. These are hypotheses, not forced correctness rules. Never enforce a minimum classical contribution just to manufacture hybridity. Tune at most 12 logged rule configurations on validation, with no per-image manually chosen settings.

Compare input/output near-clipping, colour excursions and residual flat-region noise. Set bounded input-only blend-back/soft-knee parameters from validation. Account for already clipped input lights rather than treating every naturally bright pixel as a new failure. The guard runs once; no unbounded search or NIQE optimization loop. Keep sharpening off initially; only retain it if an ablation improves real detail without amplifying noise.

**Done:** weights/maps are real, finite, aligned, spatially variable on a mixed-light fixture and normalized within 1e-5; disabling a branch changes output on a fixture where its weight is nonzero. Removing quality awareness triggers a measurable difference on a highlight fixture. Visual maps come from the exact inference call, not a separate decorative computation. A deep-only result is labelled deep-only if the selector chooses it everywhere.

## 5. Data, leakage and metric protocol

1. Create a manifest with dataset, scene/group ID, low/reference paths, dimensions, SHA256, dataset licence/access terms and split. Record the model code licence and checkpoint redistribution/use terms separately; do not assume code and weights have the same rights. Verify pair correspondence and remove corrupt/misaligned pairs before freezing, with exclusions logged. Detect exact duplicates and manually inspect near-duplicates across splits; keep scene/exposure variants in the same group.
2. Audit upstream checkpoint training data. A validation split carved from its training corpus is **adaptation validation with upstream exposure**, not an independent holdout. Use a documented unseen paired set or independent scene subset excluded from all training if possible. Otherwise disclose exposure and limit the claim; do not call contaminated validation generalization.
3. For our fine-tune, reserve 10% of official training scene groups with seed 42, at least 30 groups when enough exist. This does not cure upstream exposure. Keep official tests sealed. Freeze the manifest/hash, selection rules, bins and metric settings before comparing runs.
4. Use official LOL-v2 Real test for final paired reporting if its licence/access allow; report Synthetic separately if used. Add at least 20 independent real images across indoor, outdoor, mixed-light, severe darkness and bright point sources, including at least five freshly captured team images if feasible. Unpaired examples establish observed behavior, not full-reference accuracy.
5. Inference receives only low images/config/weights. Run it without test references mounted; save all outputs. The evaluation process receives saved predictions and references afterward. Log file accesses for a small isolation audit. A successful reference-free run is an acceptance gate.

Freeze this metric protocol in a config before experiments:

| Metric | Required convention |
|---|---|
| PSNR | Per-image RGB MSE in float [0,1], full unpadded dimensions, no border shave or GT scaling; arithmetic mean of per-image dB. Identical pair is +inf and explicitly represented in serialization |
| SSIM | Pin implementation/version; RGB channel mean, Gaussian window 11, sigma 1.5, population covariance, data range 1; define valid-window boundary handling. Inputs below window size are explicit unsupported metric cases |
| LPIPS | Official package, fixed AlexNet weights/hash, eval mode, RGB mapped to [-1,1], same dimensions, lower is better |
| NIQE | Pin a validated implementation and natural-scene statistics/hash; use its documented luminance/block convention, no custom training on test. Too-small/invalid inputs yield null plus reason, never zero |
| Latency | Actual wall clock including decode, preprocessing, both branches, fusion and output encoding; GPU synchronize boundaries. Separately report model-only, cold-start and network-inclusive live request timing |
| Noise | Fixed input-derived flat-shadow mask, wavelet HH MAD / 0.6745 before/after, valid pixels/counts. Report gain-normalized noise proxy and exposure gain; do not claim calibrated sensor noise or reward black output |
| Edge | Paired-reference Sobel gradient correlation in predeclared valid structural regions, with denominator safeguards. Unpaired input/output correlation is labelled a structural proxy; flat undefined cases are null |

All methods use the same image IDs, preprocessing, resolution and scoring implementation. Benchmark native size with padding/cropping where possible; if tiling is needed use fixed overlap and blending for every relevant run and disclose it. Live resized scores are a separate table. No per-method hidden resizing, geometric alignment to GT, histogram matching or metric-dependent output selection.

Report image counts, per-image results, mean/median and 95% paired bootstrap confidence intervals (2,000 scene-group resamples, seed 42) for differences. List every failure; require 100% prediction coverage to certify an aggregate. Null metrics retain their reasons and valid counts. Inspect fixed representative, worst-PSNR and worst-LPIPS examples with identical display settings.

### Selection gate (engineering targets, not literature claims)

Choose the final hybrid on frozen validation only. Primary target versus deep-only: at least +0.20 dB mean PSNR, SSIM drop no worse than 0.005, LPIPS increase no worse than 0.005; paired PSNR-difference CI lower bound must exceed 0 to claim demonstrated PSNR improvement. Also report brightness/noise/edge and illumination-bin results.

A different tradeoff can be retained if it is declared explicitly before the final test: PSNR within 0.20 dB, SSIM within 0.005, LPIPS improvement at least 0.01, and new clipping reduced by at least 20% relative where the deep baseline has nonzero new clipping. Identify it as a tradeoff, not a PSNR win. If neither gate passes, report the hybrid as required and disclose its regression; continue tuning only within the allocated time. No fabricated improvement and no deep output renamed hybrid. The PDF permits adaptive selection; a selector may choose deep for some inputs.

Freeze code/config/checkpoint hashes at H+13 and evaluate the official test once as a selection-free reporting pass. Infrastructure reruns use the same frozen artifacts and a reason log. Any algorithm change after viewing test results invalidates the untouched-test claim and must be disclosed.

## 6. Experiments and evidence artifacts

| ID | Run | Question |
|---|---|---|
| E0 | Input and simple fixed gamma sanity baselines | Are we doing more than brightening? |
| E1 | Fixed classical | Classical starting point |
| E2 | Adaptive classical | Do measured image characteristics help? |
| E3 | Pretrained deep, GT correction disabled | Deployable model baseline |
| E4 | Fine-tuned deep | Does actual adaptation help? |
| E5 | 50/50 classical/deep | Does spatial adaptivity beat trivial blending? |
| E6 | Illumination-only fusion | Contribution of illumination guidance |
| E7 | Illumination/noise/structure fusion | Contribution of additional guidance |
| E8 | E7 plus quality guard | Contribution of final quality awareness |

Keep backbone fixed within fusion ablations. Required comparison rows are E2, the selected deep checkpoint, and E8. If E4 is unavailable, record its status, not a numeric row. Learned fusion, FiLM, large ensembles, ONNX and torch.compile are deferred under this deadline; only revisit after all release gates pass.

Each run directory contains resolved config, code revision plus dirty patch hash, environment lock, seed, split hash, checkpoint hash/provenance, command, timestamps, stdout/stderr, parameter count, hardware/precision, memory, saved predictions, per-image metrics and generated summary. Large data/checkpoints live outside Git with a download manifest. No expectation values enter result fields.

Proposed implementation interface (not currently existing commands): `python -m p1.audit_data`, `python -m p1.train`, `python -m p1.infer`, `python -m p1.evaluate`, `python -m p1.report`, `python -m p1.acceptance`. Each accepts a versioned config; evaluation and reporting never run training. Paths live in config, not hardcoded machine directories. Keep modules for data, analysis, classical, model adapter, fusion, metrics and app; one inference service serves both CLI and UI.

**Done:** a clean environment can reproduce one full comparative report from real artifacts; report cells trace to per-image records; at least E1/E2/E3/E5/E6/E7/E8 are implemented and compared; at least three real failure cases include input, all relevant outputs, identical crops, cause and limitation. A run that does not exist remains NOT RUN.

## 7. 18-hour critical path

Hours are elapsed from implementation start, not promises of training duration. The project owner owns Part A, Part C and evaluation; the teammate owns Part B, then the app/deployment. Name the actual people in the run log at H0. Use the same frozen validation manifest for tuning all branches and the separate held-out test only after the H+13 freeze.

| Window | Work | Definition of done / cutoff |
|---|---|---|
| H0-1 | Owner starts Part A; teammate probes Kaggle and selects a deployment host; freeze interface and name both people in run log | Both acknowledge interface; GPU/quota/data and model licence checked; deployment host chosen |
| H1-3 | Part A pipeline; Part B pretrained inference and training pilot; owner starts metrics/manifest; teammate runs host hello-world | Five handoff images per branch; full-frame memory measured; public hello-world URL responds by H+3 |
| H3-5 | Part B exports first checkpoint by H+4; owner completes maps and initial fusion; teammate builds app shell against shared interface | Real E1/E2/E3/E5 metrics; exported checkpoint runs outside Kaggle; app accepts a real upload |
| H5-9 | Bounded fine-tune; owner completes fusion ablations/evaluation; teammate integrates app and deploys actual service | Actual adaptation evidence; E6-E8 results; endpoint serves real output. Challenger only if it cannot delay these |
| H9-11 | Select pretrained/fine-tuned, memory/latency fixes, complete comparison | Selection rules applied; no OOM on live acceptance set; retain original candidate if tuning regresses |
| H11-13 | Robustness cases, acceptance audit, freeze | Completed manifest/config/checkpoint freeze, negative controls pass, no placeholders in release outputs |
| H13-15 | Sealed final evaluation and generated report | Full coverage, honest metrics in required order, USNs/team identity supplied by team |
| H15-17 | Public phone/laptop checks, docs, three-minute rehearsal | Live URL works, architecture matches implementation, every member explains contribution and failure |
| H17-18 | Submission buffer and final availability check | All required links/artifacts submitted; no new architecture/training changes |

When dataset/checkpoint access fails, exhaust official compatible alternatives within H+1; do not spend the whole deadline on downloads. If Kaggle and local fine-tuning both fail, ship the working Part A branch plus a verified pretrained deep comparator and adaptive hybrid if that checkpoint runs; mark adaptation BLOCKED and document why. If even a pretrained checkpoint is unavailable, ship Part A alone with a clearly failed Part B/hybrid requirement and no invented comparison. A locally trained tiny substitute is a separately named baseline, never HVI. If deployment CPU is too slow, select an accessible GPU host or lower declared live resolution based on measurements. Kaggle is for Part B training, not assumed durable public hosting.

## 8. Release gates

Start live mode at max long side 768, preserve aspect ratio and show processed dimensions. Raise to 1024 only after measured headroom. Handle EXIF orientation, grayscale, RGBA with explicit compositing, corrupt/unsupported files and input limits. Never silently present a resized output as native-resolution restoration. No API key is needed for our model's inference.

Measure at least 20 varied valid live images after three warmups on declared deployment hardware: every image under 10 s, median/p95/max reported, zero crashes/OOM. Measure three cold starts separately. Perform real phone and laptop browser checks of upload, loading/error states, three outputs, maps and download. Include a judge-like fresh upload rather than presets alone. Offline metrics may finish separately; UI marks unavailable/pending diagnostics and never substitutes constants.

Benchmark AMP against FP32: no NaNs, max output absolute deviation <=0.01 and validation PSNR loss <=0.05 dB. Otherwise use FP32 or documented mixed precision. These are release tolerances, not guarantees of numerical equivalence.

Final checklist: runnable source with pinned environment; checkpoint provenance/download path; one-command startup or Dockerfile; public URL; actual architecture/design; comparative evaluation with correct filename/first-line USNs/metric order; ablations; three failure cases; privacy behavior inspected in code and logs; limitations acknowledging inferred details. No promise of forensic authenticity. Every member rehearses the model, classical choices, metric definitions and their own implementation.

**Completion means all mandatory release gates have evidence.** A documentation plan, passing mocked test, attractive screenshot, synthetic smoke dataset or successful import alone cannot complete the project. Failed optional improvements can be omitted with a record; failed mandatory deliverables remain explicit blockers.
