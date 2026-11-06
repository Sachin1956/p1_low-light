# Supplementary low-light dataset research

## Recommendation

Use **SMALL v1.0.0 (Smartphone Multi-Device Low-Light Dataset)** as the supplementary dataset for P1. It is a distinct dataset, not an extension or merge of the existing project set (LOL-v1, LOL-v2 Real/Synthetic, MILL small DSLR/smartphone, and the FiveK subset). Keep its directory, split manifest, metrics, and Drive folder separate.

The official Mendeley Data record describes 504 paired low-light/brighter-reference images from 371 indoor scenes, captured on five smartphone models as rendered sRGB. It supplies scene IDs, device/geometry metadata, severity labels, and a recommended scene-disjoint split of 403 train, 51 validation, and 50 test pairs. The stated license is CC BY 4.0. These characteristics complement the existing mostly fixed-camera benchmark coverage with mobile-camera and cross-device variation, while its paired references support PSNR/SSIM/LPIPS on the designated paired test set. [Official dataset record](https://data.mendeley.com/datasets/vnz4gcw8wk/1)

The official release is listed as `SMALL_v1.0.0.zip`, approximately 4.69 GB. Preserve the maintainers' split exactly; do not create a random alternative split or tune on test. The download package is large but manageable for Drive and substantially smaller than full FiveK or SID downloads.

## Alternatives considered

| Dataset | Complementary value | Why it is not the selected first supplement |
|---|---|---|
| **SID (Sony subset)** | RAW short-exposure/long-exposure pairs for extreme low-light sensor restoration; official site lists a 25 GB archive and official train/validation/test lists. [Official code/data instructions](https://github.com/cchen156/Learning-to-See-in-the-Dark), [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Learning_to_See_CVPR_2018_paper.html) | Excellent future RAW-specific extension, but far larger and requires a RAW pipeline; less aligned with the current RGB HVI-CIDNet application. |
| **LSRW** | 5,650 real low/normal pairs across Nikon and Huawei cameras, including indoor/outdoor capture. [Paper](https://arxiv.org/abs/2106.14501) | Strong alternative, but the source project is less straightforward for a license-verified, reproducible team package than SMALL's versioned Mendeley record. |

These are alternatives for later work; they are not included in this supplementary package. No images from the previous project dataset are copied into or merged with this one.

## Usage and split rules

1. Use SMALL only as a separate supplementary benchmark and report its results in a separate table from the existing project datasets.
2. Keep all scene-group pairs in the official train/validation/test partition supplied by SMALL. Do not re-randomize or move test scenes.
3. ML fine-tuning, if undertaken, may use only the official training split. Use official validation for checkpoint and hyperparameter selection. Evaluate the official test split once after freezing the pipeline.
4. Classical CV and hybrid methods may run on all inputs for inference; tune their parameters on training/validation only. Do not use test references or aggregate test metrics to tune.
5. Use PSNR, SSIM, and LPIPS only with the genuine supplied paired references and only on the designated evaluation split. Report device and scene subgroup results where meaningful.
6. Attribute the dataset and preserve its CC BY 4.0 notice and citation. Keep the original license and README in any shared copy.
7. Do not merge with existing P1 data, change image color encoding, or overwrite official files. Any derived/resized copies must be stored separately and documented.

## Citation

Do, Tuan Minh; Mai, Hoang Phuc; Phan, Cong Tam; Phan, Thi Thu Hong (2026). *SMALL: Smartphone Multi-Device Low-Light Dataset*, Version 1.0.0. Mendeley Data. DOI: [10.17632/vnz4gcw8wk.1](https://doi.org/10.17632/vnz4gcw8wk.1).
