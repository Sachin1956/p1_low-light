# P1: low-light restoration research

Checked **2026-11-06**, using primary papers, organizer reports and official code. “Verified” below means the source supports the stated result; no models were downloaded, reproduced or timed locally. Recommendations are engineering judgments.

**Decision for 18 hours / RTX 3050 4GB:** reuse pretrained HVI-CIDNet as the main deep branch. Prioritize a working input-only baseline, honest evaluation and a packaged demo. Kaggle is optional overflow compute; 2026 challenge methods are frontier context, not required implementation targets.

## Backbone evidence and compute

| Candidate | Verified evidence | Practical interpretation |
| --- | --- | --- |
| **HVI-CIDNet, CVPR 2025** | HVI separates chromatic information from intensity; CIDNet processes these through interacting branches. The paper reports **1.88M parameters, 7.57 GFLOPs at 256×256** and explicitly uses GT-mean correction for its LOL-v1 comparison. [CVF publication](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html), [paper, Table 1](https://arxiv.org/html/2502.20272v1) | Strong first candidate for P1's color/noise requirements and limited inference budget. Paper FLOPs do not establish CPU speed, full-resolution memory or a sub-10-second service response. |
| **Retinexformer, ICCV 2023** | One-stage illumination estimation and illumination-guided restoration. Original paper reports **1.61M parameters, 15.57 GFLOPs**, with complexity measured at **256×256**. [CVF publication](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.html), [paper, Table 3](https://arxiv.org/html/2303.06705v1) | Useful independent baseline; modest weight count, but greater reported arithmetic than CIDNet. Profile the exact implementation: papers can differ in counting conventions/configurations. |

The official [CIDNet repository](https://github.com/Fediory/HVI-CIDNet#weights-and-results-) lists dataset-specific weights and some **lost** checkpoints. Its LOL-v1 `w_perc.pth` reports PSNR/SSIM **23.8091/0.8574 without GT mean**, versus **27.7146/0.8760 with GT mean**. These are different evaluation protocols, not evidence of improved deployment inference. The official [Retinexformer repository](https://github.com/caiyuanhao1998/Retinexformer#3-testing) links pretrained models and dataset-specific test commands. Download links were located; archive contents, availability and hashes were not audited.

## Challenge results: scope matters

| Event / task | Verified result | Qualification |
| --- | --- | --- |
| **NTIRE 2025 LLIE** | **NWPU-HVI first overall**: PSNR **26.24**, SSIM **0.861**, LPIPS **0.128**, NIQE **10.95**. Imagine has higher PSNR, **26.35**, but ranks second overall. | Winner is **FusionNet: CIDNet + Retinexformer + ESDNet**, not either standalone backbone. Composite ranking differs from PSNR ranking. [Organizer report, Table 1 and §3.1](https://arxiv.org/html/2510.13670v1), [CVF proceedings record](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Liu_NTIRE_2025_Challenge_on_Low_Light_Image_Enhancement_Methods_and_CVPRW_2025_paper.html). |
| **NTIRE 2026 LLIE / joint denoising** | Track 1 reference winner **SYSU-FVL**: **20.71 dB / 0.6773 SSIM / 0.4802 LPIPS**; non-reference winner **AAIR-LAB**. Track 2 joint denoising reference winner **BAU-Vision**; non-reference winner **DH-XHDL-Team**. | SYSU-FVL fuses **ESDNet and DarkIR**. These reference/perceptual tracks have different winners; scores are not comparable to 2025 or LOL. [Organizer report, Table 1 and §3.1](https://arxiv.org/html/2604.17669v1). |
| **NTIRE 2026 efficient LLIE** | **MiVideo first**, using **RetinexFormerRefine**, **927,049 parameters**; SSIM **0.5654**, LPIPS **0.3632**. | Modified architecture, not the original ICCV checkpoint. Report describes smartphone images at **3024×4032**, confidential test GT and training on **8 A100 GPUs**. Small inference model does not imply cheap training. [Organizer report, Table 1 and §3.1](https://arxiv.org/html/2605.02212v1). |
| **NTIRE 2026 Twilight Cowboy** | Corrected final leaderboard: **MiAlgo first**, **40.62 dB / 0.9875 SSIM**. | Joint alignment/denoising of low-light captures with hand tremor; separate burst restoration task. Use the corrected **March 21** results, not March 20's provisional table. [Official challenge site](https://www.nightimaging.org/). |

These are verified **challenge-specific leading results**, not a verified universal “current SOTA” ordering. Neither publication-era SOTA language nor repository winner badges establish that an unchanged public checkpoint wins P1. No matched-protocol evidence was found establishing the proposed classical/CIDNet fusion as SOTA.

## Checkpoint leakage and test-time GT exposure

- **Explicit contaminated checkpoint:** CIDNet's README documents training on the LOL-v1 test set and distributes `LOLv1/test_finetuning.pth`, while discouraging this practice. Exclude it from an independent LOL-v1 benchmark. Do not infer that every other official checkpoint is contaminated; their exact training histories still need checking. [Official disclosure](https://github.com/Fediory/HVI-CIDNet#weights-and-results-).
- **Oracle exposure correction:** CIDNet's [`measure.py`](https://github.com/Fediory/HVI-CIDNet/blob/master/measure.py) applies `--use_GT_mean` by multiplying the prediction by `mean(gray(GT))/mean(gray(prediction))`, then clipping. Retinexformer's [`test_from_dataset.py`](https://github.com/caiyuanhao1998/Retinexformer/blob/master/Enhancement/test_from_dataset.py) has analogous `--GT_mean` correction. Both flags are opt-in. This modifies predictions using test labels; it is distinct from ordinary GT access solely to calculate metrics. Disable both for deployment-valid evaluation and label any oracle experiment separately.
- **Split provenance:** The 2026 LLIE report explicitly incorporates paired validation images into SYSU-FVL's final training. That validation split cannot remain an independent holdout afterward. This alone does not demonstrate hidden-test leakage. [Report, Appendix B](https://arxiv.org/html/2604.17669v1).
- **Ambiguous selection language:** The efficient-track report says MiVideo selects the best model on “validation/test sets,” despite describing hidden test GT. This warrants clarification of model selection and leaderboard feedback; the wording alone does **not** prove prohibited GT access. [Report §3.1](https://arxiv.org/html/2605.02212v1#S3.SS1).

## Recommendations for P1

1. Start with a documented **non-test-finetuned CIDNet checkpoint**. Use `LOLv1/w_perc.pth` as a concrete official starting point for a LOL-v1 baseline; evaluate cross-domain behavior separately. Compare against Retinexformer only after HVI inference, evaluation and the demo work. Record source commit, weight URL, SHA-256, architecture, training datasets/splits and every exposure/ensemble setting. Treat unknown provenance as unresolved, not clean by default. [Official checkpoint table](https://github.com/Fediory/HVI-CIDNet#weights-and-results-).
2. Freeze gamma, checkpoint choice and classical/fusion parameters on a separate development split. Lock test images away from training and selection; score final saved predictions without GT-based rescaling. Report RGB range, resizing/cropping, SSIM implementation and LPIPS backbone consistently.
3. Benchmark original, classical-only, deep-only and fused outputs on the same untouched paired set, plus unseen noisy/color-cast scenes. Retain fusion only if measured gains justify latency and visible artifacts; NIQE alone cannot establish faithful detail recovery.
4. Begin with one network and no test-time ensemble. Retinexformer's official script supports **eight transformed forward passes**, increasing compute; the 2025 winning fusion also requires three models. [Inference code](https://github.com/caiyuanhao1998/Retinexformer/blob/master/Enhancement/test_from_dataset.py), [2025 report](https://arxiv.org/html/2510.13670v1). Measure warmed end-to-end median/p95 latency and peak memory at target resolution on the actual CPU/GPU, including classical processing and encoding. Evaluate tiled inference for seams if memory requires it.
5. Treat reported patch FLOPs as screening evidence only. Prefer pretrained inference over challenge-scale retraining for the hackathon; reserve newer ensembles or generative methods for a demonstrated failure that simpler baselines cannot address. Neither deterministic inference nor a non-diffusion architecture guarantees authentic recovery of missing texture.

### Time and memory budget (proposed, not measured)

| Hours | Deliverable / stopping rule |
| --- | --- |
| 0–3 | Obtain official HVI code/weights, record provenance, run batch-one inference and verify color/channel handling. Establish FP32 outputs at 400×600 or a smaller development resolution. Stop architecture exploration. |
| 3–7 | Run paired evaluation without GT correction; add required classical baseline and failure examples. Keep test labels outside inference. |
| 7–11 | Integrate the demo and simple fusion; retain deep-only as a selectable baseline. Tune on development images only. |
| 11–15 | Profile target resolutions, peak VRAM and end-to-end latency. If necessary, evaluate overlapping 256/384-pixel tiles with feathered blending; check seams and global exposure consistency. Try mixed precision only after comparing against FP32 for numerical/visual regressions. |
| 15–18 | Freeze settings, finish ablations, package dependencies/checkpoint instructions and rehearse the demo. Retinexformer comparison is stretch work; skip retraining and multi-model ensembles. |

Batch one, inference mode and sequential classical/deep processing are reasonable starting choices for **4GB VRAM**, but no full-resolution fit or latency guarantee has been established. Small parameter counts do not bound activation memory. Disclose any resizing or tiling in both quality and speed results.

Use **Kaggle only if local setup or evaluation becomes a bottleneck**, with a bounded setup attempt; accelerator availability, session limits and transfer time have not been verified. Any optional fine-tuning must use training/development data and leave time for regression checks. Do not make the local demo depend on a remote session.

The 2026 reports verify strong task-specific entries, but this research did not establish downloadable, matching checkpoints plus complete reproducible inference for each winner. MiVideo's eight-A100 training and the 2025 three-model fusion are reasons to reuse a simpler public backbone within this deadline, not evidence that those methods cannot run inference on a smaller GPU. Avoid spending the 18-hour window reproducing leaderboard training.
