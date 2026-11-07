"""Fetch one side of the pinned LOL-v2 Real test mirror."""

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REVISION = "cd1861c6f46f0f0c4ff308bce536eb09bedb9f74"
MIRROR = f"https://huggingface.co/datasets/pkrushith/lolv2-real/resolve/{REVISION}/Test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side", choices=("Input", "GT"))
    args = parser.parse_args()
    with (ROOT / "data_manifests/LOLv2-Real-paired.csv").open(newline="", encoding="utf-8") as source:
        ids = [row["pair_id"] for row in csv.DictReader(source) if row["official_partition"] == "test"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Expected 100 unique official test IDs.")
    folder = ROOT / "data" / "submission_lolv2" / args.side
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(pair_id):
        path = folder / f"{pair_id}.png"
        if not path.exists():
            partial = path.with_suffix(".part")
            with urlopen(f"{MIRROR}/{args.side}/{path.name}", timeout=45) as response:
                partial.write_bytes(response.read())
            with Image.open(partial) as image:
                if image.format != "PNG":
                    raise ValueError(f"Expected PNG for {pair_id}.")
                image.verify()
            partial.replace(path)
        return f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(fetch, ids))
    (folder / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(f"Pinned mirror {REVISION}: {len(hashes)} {args.side} PNGs in {folder}", flush=True)


if __name__ == "__main__":
    main()
