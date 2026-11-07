"""
Robust downloader for SMALL test set pairs from Google Drive.
Handles Google Drive large file virus-scan confirmation prompts and rate limits.
"""

import csv
import hashlib
import re
import time
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "SMALL_v1.0.0" / "splits"
TEST_TXT = SPLITS_DIR / "test.txt"
OUT_DIR = REPO_ROOT / "data" / "SMALL_v1.0.0"
MANIFEST_PATH = REPO_ROOT / "data_manifests" / "SMALL-v1.0.0-test.csv"


def download_gdrive_file(session: requests.Session, file_id: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True  # Already downloaded

    url = "https://drive.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True, timeout=30)

    # Check for confirmation token
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if not token and response.text:
        match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
        if match:
            token = match.group(1)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(url, params=params, stream=True, timeout=30)

    if response.status_code == 200:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        return True
    else:
        print(f"Failed with status {response.status_code} for {file_id}")
        return False


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    import gdown

    test_ids = set(TEST_TXT.read_text().strip().splitlines())
    drive_url = "https://drive.google.com/drive/folders/1PF674712mdPsa2XbDHal9lMiVoT0Mk8f"
    items = gdown.download_folder(url=drive_url, output=str(OUT_DIR), skip_download=True)

    to_download = []
    for item in items:
        p = Path(item.path)
        base = p.name
        stem_prefix = "_".join(base.split("_")[:2])
        if stem_prefix in test_ids:
            dest_file = OUT_DIR / p
            to_download.append((item.id, dest_file, stem_prefix, p.parent.name))

    print(f"Total test set files to acquire: {len(to_download)}")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    success = 0
    for idx, (file_id, dest, stem, category) in enumerate(to_download, 1):
        if dest.exists() and dest.stat().st_size > 1024:
            success += 1
            continue

        print(f"[{idx}/{len(to_download)}] Downloading {dest.name}...")
        ok = download_gdrive_file(session, file_id, dest)
        if ok:
            success += 1
        time.sleep(0.5)

    print(f"Downloaded/Verified {success}/{len(to_download)} files.")

    pairs = {}
    for _, dest, stem, category in to_download:
        if not dest.exists() or dest.stat().st_size <= 1024:
            continue
        if stem not in pairs:
            pairs[stem] = {}
        if "low" in dest.name:
            pairs[stem]["low"] = dest
        elif "reference" in dest.name:
            pairs[stem]["ref"] = dest

    valid_pairs = {k: v for k, v in pairs.items() if "low" in v and "ref" in v}
    print(f"Verified complete paired instances: {len(valid_pairs)} / {len(test_ids)}")

    rows = []
    for pair_id in sorted(valid_pairs.keys()):
        p = valid_pairs[pair_id]
        low_p = p["low"]
        ref_p = p["ref"]
        rows.append({
            "pair_id": pair_id,
            "dataset": "SMALL_v1.0.0",
            "split": "test",
            "low_path": str(low_p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "low_sha256": compute_sha256(low_p),
            "reference_path": str(ref_p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "reference_sha256": compute_sha256(ref_p),
        })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "dataset", "split", "low_path", "low_sha256", "reference_path", "reference_sha256"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} verified pairs to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
