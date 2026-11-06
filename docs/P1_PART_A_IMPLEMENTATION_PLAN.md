# P1 Part A: classical implementation plan

Status: PLANNED. This revision addresses the review brief that temporarily replaced the earlier local draft. No method or result is implemented. Read the official P1 PDF, `P1_ML_EXECUTION_PLAN.md`, `P1_AGENT_CONTRACT.md`, and `p1_part_a_research.md` alongside this plan.

## Ownership and one 18-hour clock

H0 is implementation start; H18 is submission. The project owner owns Part A, Part C integration and evaluation. The Kaggle teammate owns Part B and, after exporting a checkpoint at H+4, the app and deployment. Write actual names and H0 time in the run log. Fine-tuning and deployment overlap in H5-9; training stops after four elapsed training hours or H+10. Freeze all choices at H+13, score held-out test at H13-15, then rehearse and submit at H15-18. These are the same milestones as the main ML plan. If Part B training competes with deployment, prioritize the live app.

Part A's goal is to improve visibility while limiting shadow noise, halos, clipped lights, colour drift and detail loss. Its falsifiable hypothesis is that input-adaptive settings (E2) outperform the *same operators* with fixed settings (E1) across varied illumination. No result is claimed yet.

## Gate 0: data and interface

The committed dataset protocol and manifests use 20% group validation with seed 1956; the main ML plan is now aligned to that split. Use the existing manifests only after auditing and share the exact hash with Part B and Part C. The manifests list files, not verified pairs. Build a pair index with one low image, one aligned reference, original partition, scene group and hashes. Count pairs, not rows. Keep official test sealed. LOL-v2 Real validation is primary; MILL-small validation probes intensity and camera behavior; SMALL validation is optional if files are available and verified. FiveK is a tonal probe only.

Use the shared `enhance(rgb_uint8, config) -> EnhancementResult` boundary. Return same-size oriented sRGB uint8 image, aligned HxW float32 maps, actual selected parameters, timing and config version. The caller handles EXIF/decode once. Inference receives no reference. Both branches use the same five handoff images. The project owner, as integrator, checks matching IDs and dimensions.

**Done:** audited pair index and shared split hash; five reference-free handoff images; pinned same-host settings yield identical outputs or a documented tolerance of one 8-bit level.

## Gate 1: finish E1 before E2

First save E0 identity/fixed gamma. Then deliver E1 end to end: max-RGB illumination, guided refinement, bounded Retinex gain, guided/bilateral denoise, luminance-only CLAHE, conservative colour correction and highlight protection. Internally use float32 RGB [0,1]. Approximate illumination by `T0=max(R,G,B)`; refined T gives a Retinex proxy `J=clip(I/max(T,tau) * T^gamma,0,1)` subject to a separate gain cap and smooth local highlight blend. This is LIME-inspired, not LIME's exact solver or physical radiance recovery. Use the same processing code for E1 and E2; E1 supplies fixed values.

Profile E1 at 768 px long side on five images including flat shadow and point light, then estimate full validation runtime. Try OpenCV contrib's guided filter; if unavailable, use a separately labelled bilateral fallback or tested local guided implementation. Never silently substitute. Save full JSONL provenance and predictions before noise-adaptive work begins.

**Done:** E1 satisfies every mandatory classical operation, has finite same-size outputs, and reaches Part C as real saved images before E2 starts.

## Gate 2: E2's concrete monotone mapping

Compute input sRGB diagnostic luminance `Y=0.2126R+0.7152G+0.0722B`. Features: `m=median(Y)`, `h=fraction(Y>=0.90)`, and `n=flat-shadow high-pass/wavelet MAD noise proxy`. Let `q` be the fraction of pixels in valid non-clipped, low-gradient shadow patches. If `q<0.02` or too few separate patches exist, mark noise invalid and use E1's fixed denoise/CLAHE values. Texture and JPEG artifacts can corrupt n, so inspect the validity mask.

Define `c(x)=min(1,max(0,x))` and initial trial mappings:

| Feature and parameter | E2 rule | Intended behavior |
|---|---|---|
| Median luminance to gamma | `d=c((0.55-m)/0.45)`; `gamma=1-0.40*d` | Darker image gets stronger bounded illumination lift; gamma stays 0.60-1.00 |
| Highlight fraction to gain cap | `gain_max=2+3*d*(1-c(h/0.10))` | More existing highlights reduce maximum gain; cap stays 2-5 |
| Valid noise to denoise | `z=c((n-0.01)/0.05)`; `strength=s_min+(s_max-s_min)*z` | More measured noise increases denoising within library-specific validated bounds |
| Valid noise to CLAHE | `clip_limit=2.2-0.8*z` | More measured noise reduces local contrast amplification; OpenCV clip stays 1.4-2.2 |

Keep CLAHE grid fixed at 8x8 initially; apply it to luminance only, and blend down its effect in strong highlights/noisy areas. `s_min/s_max` and any changes to the trial thresholds are versioned validation choices, included in at most 12 logged complete configurations. They are not paper-derived optima. The spatial highlight mask and structure protection are identical in E1/E2. No filename, reference or test statistic influences the mapping. A controlled sweep must show monotonic outputs and at least two real images must select different parameters.

**Done:** finite, aligned maps; logged input features and selected parameters; invalid-noise fallback works; E2 differs from E1 only by parameter selection; repeated input produces repeatable output.

## Gate 3: selection and ablations

Tune on frozen validation only. First measure a full 768 px validation pass and estimate `N_images × measured p95 seconds`; schedule at most 12 complete configurations *total*, not a Cartesian search. Native-resolution output is a separate, identically sized final comparison for shortlisted E1/E2. Run one-at-a-time ablations for illumination refinement, denoising, CLAHE, colour correction, highlight guard and adaptivity only as time permits, saving each output and config. Keep official test untouched until H+13.

**Promote E2 over E1 only if** mean paired scene-level PSNR difference is positive and the lower bound of a 95% paired bootstrap interval (2,000 scene-group resamples, seed 42) is above zero; mean SSIM drop is no worse than 0.005 and LPIPS increase no worse than 0.005. Pin LPIPS AlexNet weights and RGB [-1,1] input. Also require no severe new clipping/halo on the fixed sentinel set. Measure edge-band luminance overshoot/undershoot near declared strong input edges, plus have two named reviewers inspect identical crops without method labels and record disagreements. If E2 is statistically indistinguishable or worse, select E1 for production and report E2 honestly as an ablation; maps can still explain the attempted adaptation, but cannot support an improvement claim.

Score identical image IDs with RGB PSNR/SSIM/LPIPS; NIQE and noise/edge estimates are separately labelled diagnostics. Report coverage, invalid counts, clipping, latency, per-image results and scene-group summaries. MILL levels share scenes, so resample scenes rather than levels for intervals. A person other than the evaluator author independently recomputes PSNR on five saved pairs. Document at least three actual failures with fixed-scale crops.

**Done:** E0/E1/E2 predictions and chosen ablations cover declared validation IDs or explicitly fail coverage; selection follows the preregistered rule; every table cell regenerates from saved predictions.

## Gate 4: integration, timing and demo

The project owner integrates Part A maps and images with Part B's real exported checkpoint. The teammate builds and hosts the app after H+4. Show original, classical, deep and hybrid outputs with maps and parameters from the same call. A missing Part B is visibly labelled, never replaced with a fake deep output. Preset two or three varied verified images for a reliable explanation, then accept a fresh judge upload. If it fails, show the failure and cause.

Target Part A <=1 second at 768 px on the actual host, measured across 20 varied uploads; record median, p95 and maximum including decode/encode. Keep native-resolution benchmark results separate from resized demo results. The complete hosted response must meet the main plan's acceptance gate.

Schedule: H0-1 pair audit/interface; H1-3 E0/E1 and host hello-world; H3-5 E2 features and H+4 checkpoint handoff; H5-9 bounded tuning/ablations while teammate builds app; H9-11 integration and profiling; H11-13 freeze; H13-15 held-out evaluation; H15-18 demo and submission. If behind, ship E1 plus genuine Part B comparator and working app before additional operators. No test-informed tuning can retain an untouched-test claim.

**Final done:** the official Part A operations run; split/pair index is audited; E1 and E2 predictions and maps are reproducible; E2 is chosen only on evidence; ablations and failures are real; shared interface and latency are measured on unseen inputs. Until then, mark status PLANNED, IMPLEMENTED-UNVERIFIED or BLOCKED.
