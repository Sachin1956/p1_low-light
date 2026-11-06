# Dataset inventory, splits, and usage rules

This document defines the dataset protocol for P1, **Robust Low-Light Image Restoration and Enhancement**. It follows Section 7 of `P1_recommended_modified_plan.md` and the hackathon requirements cited there. The plan names datasets and split rules, but it does **not** give fixed split percentages, seed, image counts, or official partition details for every dataset. The reproducible random procedure below fills in the missing local split policy without overriding official partitions.

## Dataset inventory and branch usage

| Dataset | Plan role | Classical CV | ML (HVI-CIDNet) | Hybrid / evaluation |
|---|---|---|---|---|
| **LOL-v2 Real** | Primary paired training/fine-tuning and evaluation resource | Run classical pipeline on low-light inputs; tune parameters using training/validation only | Released checkpoint inference; optional fine-tuning on training pairs only | Run fusion on the same inputs; paired metrics against aligned normal-light references. Final official test evaluation only |
| **LOL-v2 Synthetic** | Additional training diversity and separate evaluation benchmark | Run/tune on low-light inputs in train/validation | Optional fine-tuning on training pairs; validation for model selection | Evaluate separately, including final official test only after all decisions are frozen |
| **LOL-v1** | Additional benchmark comparison | Inference; validation may be used for parameter tuning if a documented random split is created from its training partition | Inference; do not fine-tune on its test partition | Compare branch and hybrid outputs against paired references when genuine aligned pairs exist; report separately |
| **MIT-Adobe FiveK** | Optional supplementary tonal/color experiment | Optional exploratory tonal/color work | No model-training use specified | Do not treat expert-retouched targets as paired low-light ground truth or report full-reference low-light restoration scores against them |
| **MILL / unpaired test images** | Optional independent generalization evaluation | Inference only | Inference only | Inference only; use no-reference/diagnostic metrics, not PSNR/SSIM/LPIPS absent genuine reference pairs |

The classical pipeline has no learned fitting stage in the plan. It may use training data for development and validation data for parameter selection, but must not use test data to tune settings. The required hybrid is initially rule-based; any later learned fusion model follows the same train/validation/test boundaries as HVI-CIDNet.

## Split policy

### Datasets with official partitions

For LOL-v2 Real, LOL-v2 Synthetic, and LOL-v1, first use the dataset's published train/test assignment. **Never randomly repartition or mix its official test data into development data.** Randomly select validation samples only from the official training portion. This is the plan's explicit rule: validation comes only from official training data, and official test sets remain untouched until final evaluation.

### Datasets without a suitable official partition

For FiveK (if used) and any locally assembled unpaired MILL/generalization set, create a random, reproducible partition by unique source image/scene, not by crops or augmented variants. These are supplemental development/diagnostic sets, not replacements for the paired benchmark test sets. FiveK can be held out for its tonal/color experiment; unpaired images are for final robustness diagnostics and must not become model-training data under the current plan.

### Random validation selection

Use a **20% validation holdout from each official training partition**, selected randomly with seed **1956**, after deduplication/grouping by source scene or capture when identifiers are available. The remaining 80% is the training/development pool. For small partitions, record exact counts and use a deterministic rounded count (`ceil(0.20 × N)` validation items, at least one if N ≥ 2); with N=1, keep the sole item in training and note that validation is unavailable. Pair files stay together. Do not split low/normal pairs, video frames from the same sequence, burst frames, crops, or augmented copies across partitions.

This 80/20 rule is a project protocol choice because the source plan requires a random validation split but does not specify its proportion or seed. Store the generated manifest with dataset-relative paths, group/pair IDs, split, seed, and source partition; commit the manifest only if dataset licenses permit, otherwise commit the generation script and a checksum/summary.

| Data split | Permitted use |
|---|---|
| Train (80% of official train) | Optional HVI-CIDNet fine-tuning; optional later learned-fusion training; exploratory classical development |
| Validation (20% of official train) | Tune classical parameters and rule-based fusion; choose checkpoints/hyperparameters; compare branch and hybrid ablations |
| Official test | One final benchmark evaluation after code, parameters, checkpoints, and model choices are frozen; no tuning, checkpoint selection, or fine-tuning |
| Unpaired generalization set | Inference and no-reference diagnostics only; not training or hyperparameter tuning if designated as final robustness evaluation |

Report per-dataset counts and exact split provenance with every metric. Keep results for each dataset separate; do not merge scores across datasets into an unlabeled aggregate.

## Hard rules

1. Preserve every official test partition exactly as supplied; do not sample from it for train or validation.
2. Make validation only from official training data for datasets that provide official partitions.
3. Apply the same source-image split to classical, deep, and hybrid comparisons. Paired low-light input and reference always remain together.
4. Prevent leakage by grouping all frames, crops, alternate exposures, and augmentations from one source scene/capture into one split.
5. Fine-tune HVI-CIDNet only on training pairs. Use validation for checkpoint/model selection. Do not fine-tune on LOL-v1 test or any benchmark test set.
6. Initially use the released HVI-CIDNet checkpoint for inference. Fine-tuning is optional and resource-dependent, as stated in the plan.
7. Keep the initial fusion rule-based. If a learned fusion model is later trained, fit it on training data and select it on validation data only.
8. Classical operations at evaluation time may process the test input, but no test reference or test-set aggregate may influence parameter choices.
9. Use PSNR, SSIM, and LPIPS only when a genuine aligned reference exists. For unpaired inputs, report NIQE, noise suppression, edge preservation, and latency as appropriate.
10. Do not claim FiveK expert retouches are paired low-light ground truth; use FiveK only for the optional tonal/color experiment described in the plan.
11. Record dataset version/source, official partition, random seed, item/group counts, preprocessing, and any excluded/corrupt files. Do not silently change a split after seeing its scores.
12. Freeze and document the manifests before final test evaluation. If a manifest must change for a documented data-integrity reason, regenerate from the same rules and disclose the change before reporting results.

## Required manifest fields

At minimum, each generated split manifest should record:

```text
dataset, dataset_version, relative_input_path, relative_reference_path,
source_group_id, official_partition, project_split, random_seed,
preprocessing_version, exclusion_reason
```

For each evaluation table, state dataset and split, pair count, metrics, input resolution, hardware, and whether the deep branch used released or fine-tuned weights. The mandatory `eval_P1_T<TeamID>.md` should follow the exact metric order required by the plan and identify split provenance.

## Scope and verification note

This document defines the split protocol; it does not download datasets or fabricate file-level manifests. Before generating manifests, verify the downloaded dataset version, official split files, pair alignment, and license/access terms. The referenced plan identifies LOL-v1's official link as being in the hackathon datasheet, and describes FiveK as listed there; obtain those exact source details from the supplied PDF/datasheet when preparing data. Dataset-specific official counts are intentionally not asserted here because they are not specified in the implementation plan's dataset table.
