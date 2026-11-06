# Dataset storage and teammate sharing

Keep image archives and extracted datasets out of this Git repository. The repository should contain download/setup instructions, license/source records, preprocessing code, and compact split manifests only.

## Current local data

Datasets are stored outside the repository under `../cv-hackathon-data/` on the maintainer workstation. Each teammate should create a local `data/` directory (ignored by Git) and download the datasets from their official source, subject to that source's terms. Do not upload or redistribute FiveK or other dataset files unless the applicable license expressly permits it. For gated releases, each teammate should accept the terms and obtain their own access.

## Recommended sharing workflow

1. Share the Git repository for code, scripts, documentation, and split-generation rules.
2. Have each teammate obtain benchmark data from the official release. This avoids license conflicts and makes provenance clear.
3. If the team needs one shared copy, use institution-approved storage (Google Drive/OneDrive/Dropbox or a campus server) with restricted access and enough quota. Share access only with teammates permitted by the dataset terms.
4. Do not use Git LFS as the default for multi-gigabyte datasets: it still incurs storage/bandwidth quota and does not resolve redistribution restrictions.
5. Publish a manifest with dataset version, official source, relative paths, official partition, project split, pair/group ID, and seed. Include SHA-256 checksums for downloaded archives so teammates can verify identical copies.
6. Keep data private; do not commit credentials, private links, access tokens, or raw data to the public repository.

## Upload the complete P1 dataset folder to Google Drive

A curated upload folder was assembled locally at:

```text
C:\Users\rohit\Documents\ChatGPT\P1-datasets-upload
```

It contains all extracted datasets (LOL-v1, LOL-v2 Real, LOL-v2 Synthetic, both MILL small subsets, and the 60-pair FiveK subset), all split/archive manifests, and README/license/sharing documents. It excludes duplicate source archives and extraction utilities. Its current size is about **5.62 GB**. Upload this folder as a whole; you do not need to upload the original `cv-hackathon-data` working directory.

The team Drive folder supplied by the user is: [P1 datasets](https://drive.google.com/drive/folders/1JSfSxRgmev0vXgvEvPKc89KrqfRRKKNL?usp=sharing). The Drive folder is not accessible to this workspace for independent verification. The user confirmed the upload; use Drive as the shared copy and download only the subsets needed for current work. The local upload folder and working dataset copies have been removed after preserving manifests in `data_manifests/`.

## Separate supplementary dataset package

The supplementary dataset is SMALL v1.0.0 and is **separate** from the original P1 dataset folder. Research and usage rules are documented in `SUPPLEMENTARY_DATASET_RESEARCH.md`. The user's supplementary Drive folder is:

[P1 supplementary SMALL dataset](https://drive.google.com/drive/folders/1PF674712mdPsa2XbDHal9lMiVoT0Mk8f?usp=sharing)

The user confirmed that the upload completed. Local image/archive copies have been deleted. Lightweight attribution, metadata, official split lists, and SHA-256 manifest are retained in `data_manifests/SMALL-v1.0.0/`. Do not merge this Drive collection with the original `P1-datasets` Drive folder. Use its official splits and report metrics separately.

1. Open [Google Drive](https://drive.google.com/) and create a folder such as `P1-datasets`.
2. Choose **New → Folder upload** and select `C:\Users\rohit\Documents\ChatGPT\P1-datasets-upload`.
3. Wait for Drive to report that the upload finished. Check the folder structure and verify a sample of archive/checksum manifests.
4. Right-click the Drive folder → **Share**. Keep **General access: Restricted**, add teammates by account email, and send them the folder link.
5. Keep the dataset README and license files alongside the data. Dataset-specific terms govern teammate access and use.

The earlier smaller FiveK-only workflow remains:

The prepared FiveK subset is about 3.71 GB decimal and includes 60 DNG/Expert A pairs, the metadata manifest, checksums, and both official license files. Its MIT/Adobe license permits research redistribution subject to the license terms; keep both license files with the images, restrict sharing to the project team, and do not use it commercially. The original DNG/TIFF pairs are for tonal/color experiments and are not low-light restoration ground truth.

1. Open [Google Drive](https://drive.google.com/) in your browser and create a folder such as `P1-FiveK-subset`.
2. Upload the local folder `C:\Users\rohit\Documents\ChatGPT\cv-hackathon-data\MIT-Adobe-FiveK-Subset` using **New → Folder upload**. Keep `DNG`, `ExpertA`, `fivek_subset.csv`, `LicenseAdobe.txt`, and `LicenseAdobeMIT.txt` together.
3. Wait for Drive to report the upload is complete. Confirm the uploaded folder contains 60 DNG files, 60 TIFF files, the CSV, and both licenses.
4. Right-click the Drive folder → **Share**. Set **General access** to **Restricted**, then add teammates' account emails as viewers (or editors if they need to manage the folder). Send them the Drive link.
5. Teammates should download the folder and verify file names and SHA-256 values against the manifest before use.

Do not set the folder to “Anyone with the link” unless the team and institution explicitly approve that distribution scope under the license. Google Drive is a transport/storage option; the license still governs who may receive and use the images.

## Current dataset footprint

The local `cv-hackathon-data` folder contains the MILLs-DSLR and MILLs-smartphone archives plus extracted images. Record the measured total with `Get-ChildItem ../cv-hackathon-data -Recurse -File | Measure-Object Length -Sum` (PowerShell) after each acquisition; compressed archive plus extracted copy consumes disk space twice until the archive is intentionally removed.

## Storage choice

For a student team, the best default is **official per-teammate downloads plus committed manifests**. If repeated downloads are too slow or network restrictions require a shared copy, use a private institutional cloud folder with access controls, and verify the license permits sharing among the team first. Keep only the scripts/manifests and source links in GitHub.
