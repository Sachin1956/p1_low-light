# P1 local dataset workspace

This directory is intentionally outside the Git checkout. It contains source archives, extracted benchmark files, and split manifests. Do not commit raw dataset files to GitHub.

## Prepared datasets

- `MILLs-DSLR/` and `MILLs-smartphone/`: extracted MILL small releases, retaining official train/validation/test directories.
- `LOL-v1/`: extracted LOL-v1 release. `our485` is its official training data and `eval15` its official test data. The manifest randomizes a validation subset from `our485` (seed 1956).
- `LOLv2-real/`: extracted LOL-v2 Real release. Official test data remains test; the manifest selects validation only from official training.
- `LOLv2_Synthetic/`: extracted LOL-v2 Synthetic mirror. Official test data remains test; the manifest selects validation only from official training. Mirror provenance is recorded in the repository documentation.
- `manifests/*.csv`: file-level project split assignments.
- `manifests/archive_checksums.csv`: archive byte sizes and SHA-256 checksums.

## Reproduce / share

Each teammate should obtain the data from the linked source under its terms, then run `python tools/prepare_datasets.py DATASET_ROOT OUTPUT.csv --seed 1956 --val-fraction 0.2` from the project checkout. Preserve official test data and all pair/scene groups. Compare archive SHA-256 values with `manifests/archive_checksums.csv`. Do not redistribute controlled/licensed imagery without permission.

## MIT-Adobe FiveK subset

`MIT-Adobe-FiveK-Subset/` contains 60 metadata-stratified DNG/Expert A pairs (about 3.71 GB decimal), selected with seed 1956 from the official catalog. Selection favors indoor, artificial/mixed-light, higher-ISO, and varied subject scenes while retaining other lighting and subject categories. It is for optional tonal/color experiments only, not paired low-light restoration ground truth. The folder includes `fivek_subset.csv`, `LicenseAdobe.txt`, and `LicenseAdobeMIT.txt`; keep these with any team-shared copy and follow the research-only license terms.

To regenerate after clearing the subset folder, run `python tools/download_fivek_subset.py DATA_ROOT/MIT-Adobe-FiveK-Subset --count 60 --seed 1956` from the project checkout. The downloader writes SHA-256 values and sizes into the CSV. For teammate sharing, use a restricted-access Google Drive folder; see `DATA_SHARING.md` in the code repository for steps.
