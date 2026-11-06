"""Create deterministic group-aware train/validation manifests for downloaded datasets.

Official test folders are never moved or re-split. The script records all files
from official training partitions and assigns scene/pair groups to train or val.
It does not redistribute copyrighted dataset files.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dng"}


def group_key(path: Path) -> str:
    """Strip common input intensity/exposure suffixes to keep a scene together."""
    name = path.stem
    name = re.sub(r"(?:[-_](?:input|low|dark))+$", "", name, flags=re.I)
    name = re.sub(r"[-_]\d+(?:[-_]\d+)?$", "", name)
    return name.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--seed", type=int, default=1956)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")

    root = args.dataset_root.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    training_files = [p for p in files if any(x in {part.lower() for part in p.parts} for x in ("train", "training"))]
    rows: list[dict[str, str]] = []
    groups: dict[tuple[str, str], list[Path]] = {}

    # Anything already in official validation/test is captured verbatim and never randomized.
    for path in files:
        parts = {part.lower() for part in path.parts}
        if "eval15" in parts:
            official = "test"
        elif "our485" in parts:
            official = "train"
        else:
            official = "test" if "test" in parts else "validation" if "validation" in parts or "val" in parts else "train" if "train" in parts or "training" in parts else "unassigned"
        # MILL maintainers explicitly identify this captured file as corrupt.
        if root.name.lower() == "mills-dslr" and path.name.lower() == "test13-9.png":
            rows.append({"dataset": root.name, "relative_path": path.relative_to(root).as_posix(), "source_group_id": group_key(path), "official_partition": official, "project_split": "excluded", "seed": str(args.seed), "exclusion_reason": "MILL maintainers report this file is corrupted after capture"})
            continue
        if official == "train":
            groups.setdefault((str(root), group_key(path)), []).append(path)
        else:
            rows.append({"dataset": root.name, "relative_path": path.relative_to(root).as_posix(), "source_group_id": group_key(path), "official_partition": official, "project_split": official, "seed": str(args.seed), "exclusion_reason": ""})

    keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n_val = max(1, round(len(keys) * args.val_fraction)) if len(keys) > 1 else 0
    validation = set(keys[:n_val])
    for key, paths in groups.items():
        split = "validation" if key in validation else "train"
        for path in paths:
            rows.append({"dataset": root.name, "relative_path": path.relative_to(root).as_posix(), "source_group_id": key[1], "official_partition": "train", "project_split": split, "seed": str(args.seed), "exclusion_reason": ""})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "relative_path", "source_group_id", "official_partition", "project_split", "seed", "exclusion_reason"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["project_split"], r["relative_path"])))
    print(f"Wrote {len(rows)} rows from {len(files)} image files to {args.output_csv}")


if __name__ == "__main__":
    main()
