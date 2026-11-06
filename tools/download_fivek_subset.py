"""Download a small, metadata-stratified MIT-Adobe FiveK subset.

Selects only original DNGs and Expert A TIFF renditions for tonal/color study.
The official dataset site and its license files remain the authority.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from lxml import html

BASE = "https://data.csail.mit.edu/graphics/fivek/"


def get_text(cell: object) -> str:
    return " ".join(cell.itertext()).strip().replace("\xa0", " ")


def catalog() -> list[dict[str, str]]:
    response = requests.get(BASE, timeout=60)
    response.raise_for_status()
    tree = html.fromstring(response.content)
    records = []
    for row in tree.xpath("//tr"):
        cells = row.xpath("./td")
        if len(cells) < 13:
            continue
        links = {a.text_content().strip(): a.get("href") for a in row.xpath(".//a")}
        dng = next((a.get("href") for a in row.xpath(".//a") if (a.get("href") or "").endswith(".dng")), None)
        tiff_a = next((a.get("href") for a in row.xpath(".//a") if "tiff16_a/" in (a.get("href") or "")), None)
        if not dng or not tiff_a:
            continue
        records.append({
            "index": get_text(cells[0]), "dng_url": dng, "expert_a_url": tiff_a,
            "subject": get_text(cells[8]).replace("Subject:", "").strip(),
            "light": get_text(cells[9]).replace("Light:", "").strip(),
            "location": get_text(cells[10]).replace("Location:", "").strip(),
            "time": get_text(cells[11]).replace("Time:", "").strip(),
            "exif": get_text(cells[12]).replace("EXIF:", "").strip(),
        })
    return records


def choose(records: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    # FiveK is not a low-light benchmark. Favor its most useful tonal/color
    # coverage: indoor/artificial/mixed-light and night/unknown scenes, while
    # retaining outdoor daylight, subject, and camera diversity.
    def score(r: dict[str, str]) -> int:
        text = " ".join((r["light"], r["location"], r["time"], r["exif"])).lower()
        s = 0
        s += 5 if any(k in text for k in ("artificial", "mixed", "night", "evening", "unknown")) else 0
        s += 3 if "indoors" in text else 0
        s += 1 if re.search(r"iso(?:\s|:)*(?:[8-9]\d\d|[1-9]\d{3,})", text) else 0
        return s
    rng = random.Random(seed)
    # Sort within useful metadata strata, randomly tie-breaking, then round-robin
    # across strata to retain category and camera variation.
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for r in records:
        key = (r["light"].lower(), r["location"].lower(), r["subject"].lower())
        buckets.setdefault(key, []).append(r)
    for bucket in buckets.values():
        rng.shuffle(bucket)
        bucket.sort(key=score, reverse=True)
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected = []
    while len(selected) < count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
    return selected


def download_file(session: requests.Session, relative: str, target: Path) -> tuple[str, int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size > 0:
        h = hashlib.sha256()
        with target.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return relative, target.stat().st_size, h.hexdigest()
    for attempt in range(10):
        if tmp.exists():
            tmp.unlink()
        h = hashlib.sha256()
        n = 0
        try:
            with session.get(BASE + relative, stream=True, timeout=(30, 240)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("content-length", "0"))
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            h.update(chunk)
                            n += len(chunk)
            if expected and n != expected:
                raise IOError(f"incomplete transfer: expected {expected}, got {n}")
            tmp.replace(target)
            return relative, n, h.hexdigest()
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == 9:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1956)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = catalog()
    selected = choose(records, args.count, args.seed)
    manifest = args.output / "fivek_subset.csv"
    results: dict[str, tuple[str, int, str]] = {}
    jobs = {}
    with requests.Session() as session, ThreadPoolExecutor(max_workers=2) as pool:
        for r in selected:
            for key, folder in (("dng_url", "DNG"), ("expert_a_url", "ExpertA")):
                rel = r[key]
                target = args.output / folder / Path(rel).name
                jobs[pool.submit(download_file, session, rel, target)] = (r["index"], key)
        for future in as_completed(jobs):
            idx, key = jobs[future]
            rel, size, sha = future.result()
            results[f"{idx}:{key}"] = (rel, size, sha)
            total = sum(v[1] for v in results.values())
            if total > 4_800_000_000:
                raise RuntimeError("Subset exceeded the 4.8 GB safety limit")
            print(f"downloaded {len(results)}/{2*len(selected)} files; {total/1e9:.2f} GB", flush=True)

    with manifest.open("w", newline="", encoding="utf-8") as f:
        fields = ["index", "subject", "light", "location", "time", "exif", "dng_path", "expert_a_path", "dng_bytes", "expert_a_bytes", "dng_sha256", "expert_a_sha256"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in selected:
            dng = results[f"{r['index']}:dng_url"]
            tif = results[f"{r['index']}:expert_a_url"]
            w.writerow({"index": r["index"], "subject": r["subject"], "light": r["light"], "location": r["location"], "time": r["time"], "exif": r["exif"], "dng_path": "DNG/"+Path(dng[0]).name, "expert_a_path": "ExpertA/"+Path(tif[0]).name, "dng_bytes": dng[1], "expert_a_bytes": tif[1], "dng_sha256": dng[2], "expert_a_sha256": tif[2]})
    for name in ("LicenseAdobe.txt", "LicenseAdobeMIT.txt"):
        response = requests.get(BASE + "legal/" + name, timeout=30)
        response.raise_for_status()
        (args.output / name).write_bytes(response.content)
    print(f"complete: {len(selected)} pairs, {sum(v[1] for v in results.values())/1e9:.2f} GB in {args.output}")


if __name__ == "__main__":
    main()
