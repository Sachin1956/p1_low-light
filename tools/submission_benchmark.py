"""Run the live restoration configuration on LOL-v2 Real test in two phases."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import _CKPT_PATH, process_image
from download_submission_split import REVISION
from p1.interface import compute_bytes_sha256, compute_file_sha256, load_oriented_rgb_uint8

DATA = ROOT / "data/submission_lolv2"
MANIFEST = ROOT / "data_manifests/LOLv2-Real-paired.csv"
EVIDENCE = ROOT / "outputs/submission"
RUN = ROOT / "outputs/tmp/submission_lolv2"
METHODS = ("classical", "deep", "hybrid")
SETTINGS = {"fusion_mode": "hybrid_guarded", "enable_quality_guard": True, "adaptive_classical": True}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def test_ids():
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        ids = [r["pair_id"] for r in csv.DictReader(f) if r["official_partition"] == "test"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("The official split must contain 100 unique IDs.")
    return ids


def infer(ids):
    if (DATA / "GT").exists():
        raise RuntimeError("Move the reference folder out of data/submission_lolv2 before inference.")
    from p1.deep_adapter import load_cidnet_model

    _, checkpoint_hash = load_cidnet_model(_CKPT_PATH, device="cpu")
    for method in METHODS:
        (RUN / method).mkdir(parents=True, exist_ok=True)
    run_meta = {
        "dataset": "LOL-v2 Real official test", "mirror_revision": REVISION,
        "manifest_sha256": compute_file_sha256(MANIFEST), "checkpoint_sha256": checkpoint_hash,
        "settings": SETTINGS, "app_sha256": compute_file_sha256(ROOT / "app.py"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_diff_sha256": compute_bytes_sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)),
        "benchmark_sha256": compute_file_sha256(Path(__file__)),
        "device": "CPU", "platform": platform.platform(), "processor": platform.processor(),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "opencv": cv2.__version__,
    }
    for _ in range(3):
        process_image(np.full((400, 600, 3), 32, dtype=np.uint8), **SETTINGS)
    records = []
    for index, pair_id in enumerate(ids, 1):
        start = time.perf_counter()
        image, image_hash = load_oriented_rgb_uint8(DATA / "Input" / f"{pair_id}.png")
        outputs = process_image(image, **SETTINGS)[:3]
        hashes = {}
        for method, output in zip(METHODS, outputs, strict=True):
            if output.shape != image.shape:
                raise ValueError(f"Unexpected resize on {pair_id}: {image.shape} -> {output.shape}")
            path = RUN / method / f"{pair_id}.png"
            Image.fromarray(output).save(path)
            hashes[method] = compute_file_sha256(path)
        records.append({"pair_id": pair_id, "input_sha256": image_hash, "prediction_sha256": hashes,
                        "decode_process_encode_ms": (time.perf_counter() - start) * 1000})
        if index % 10 == 0:
            print(f"Saved predictions for {index}/{len(ids)} test inputs", flush=True)
    write_json(EVIDENCE / "inference.json", {**run_meta, "records": records})


def score(ids):
    import lpips
    from loss.niqe_utils import niqe, to_y_channel

    evidence = json.loads((EVIDENCE / "inference.json").read_text(encoding="utf-8"))
    if [r["pair_id"] for r in evidence["records"]] != ids or evidence["manifest_sha256"] != compute_file_sha256(MANIFEST):
        raise ValueError("Prediction coverage or frozen manifest differs from the scoring split.")
    model = lpips.LPIPS(net="alex", verbose=False).eval()
    stats_path = ROOT / "external/HVI-CIDNet/loss/niqe_pris_params.npz"
    with np.load(stats_path) as stats:
        niqe_params = [stats[key] for key in ("mu_pris_param", "cov_pris_param", "gaussian_window")]

    def tensor(image):
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None] / 127.5 - 1

    def luminance(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def gradients(y):
        return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1))

    def shadow_noise(y, mask):
        hh = cv2.filter2D(y, cv2.CV_32F, np.array([[.5, -.5], [-.5, .5]], dtype=np.float32))[mask]
        return float(np.median(np.abs(hh - np.median(hh))) / .6745) if hh.size >= 64 else None

    rows = []
    for rec in evidence["records"]:
        pair_id = rec["pair_id"]
        low, low_hash = load_oriented_rgb_uint8(DATA / "Input" / f"{pair_id}.png")
        if low_hash != rec["input_sha256"]:
            raise ValueError(f"Input changed for {pair_id}.")
        reference, ref_hash = load_oriented_rgb_uint8(DATA / "GT" / f"{pair_id}.png")
        if low.shape != reference.shape:
            raise ValueError(f"Unaligned pair {pair_id}: {low.shape} and {reference.shape}.")
        y_low, grad_ref = luminance(low), gradients(luminance(reference))
        grad_low = gradients(y_low)
        flat_shadow = (y_low <= np.median(y_low)) & (grad_low <= np.percentile(grad_low, 35))
        structure = grad_ref > np.percentile(grad_ref, 70)
        sigma_low = shadow_noise(y_low, flat_shadow)
        ref_tensor = tensor(reference)
        for method in METHODS:
            prediction, digest = load_oriented_rgb_uint8(RUN / method / f"{pair_id}.png")
            if digest != rec["prediction_sha256"][method] or prediction.shape != reference.shape:
                raise ValueError(f"Changed or unaligned prediction {pair_id}/{method}.")
            mse = float(np.mean(((prediction.astype(np.float64) - reference) / 255) ** 2))
            with torch.inference_mode():
                distance = float(model(tensor(prediction), ref_tensor).item())
            niqe_value = float(niqe(to_y_channel(prediction[..., ::-1].astype(np.float32)).squeeze().round(), *niqe_params))
            y = luminance(prediction)
            sigma = shadow_noise(y, flat_shadow)
            gain = float(y[flat_shadow].mean() / y_low[flat_shadow].mean()) if sigma is not None and y_low[flat_shadow].mean() > 0 else None
            edge = gradients(y)[structure]
            edge_ref = grad_ref[structure]
            corr = float(np.corrcoef(edge, edge_ref)[0, 1]) if edge.size > 1 and edge.std() > 0 and edge_ref.std() > 0 else None
            rows.append({
                "pair_id": pair_id, "method": method, "reference_sha256": ref_hash,
                "psnr_db": -10 * np.log10(mse) if mse else "+inf",
                "ssim": float(structural_similarity(prediction, reference, data_range=255, channel_axis=-1,
                              gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False)),
                "lpips_alex": distance, "niqe": niqe_value if np.isfinite(niqe_value) else None,
                "shadow_mad": sigma, "input_shadow_mad": sigma_low, "shadow_gain": gain,
                "gain_normalized_shadow_mad": sigma / gain if gain and sigma is not None else None,
                "valid_shadow_pixels": int(flat_shadow.sum()), "edge_corr": corr,
                "new_clipping_pct": float(100 * np.mean((prediction.max(2) >= 250) & (low.max(2) < 250))),
            })
        if len(rows) % 30 == 0:
            print(f"Scored {len(rows) // 3}/{len(ids)} test pairs", flush=True)

    metrics = ("psnr_db", "ssim", "lpips_alex", "niqe", "shadow_mad", "gain_normalized_shadow_mad", "edge_corr")
    aggregates = {}
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        aggregates[method] = {}
        for metric in metrics:
            valid = [float(r[metric]) for r in selected if r[metric] is not None]
            aggregates[method][metric] = {"count": len(valid), **{
                key: ("+inf" if np.isposinf(value) else float(value)) if valid else None
                for key, value in (("mean", np.mean(valid) if valid else None), ("median", np.median(valid) if valid else None))
            }}
    times = [r["decode_process_encode_ms"] for r in evidence["records"]]
    deep = {r["pair_id"]: r for r in rows if r["method"] == "deep"}
    hybrid = [r for r in rows if r["method"] == "hybrid"]
    diffs = np.array([float(r["psnr_db"]) - float(deep[r["pair_id"]]["psnr_db"]) for r in hybrid])
    rng = np.random.default_rng(42)
    ci = np.percentile(rng.choice(diffs, size=(2000, len(diffs)), replace=True).mean(axis=1), [2.5, 97.5]) if np.isfinite(diffs).all() else None
    write_json(EVIDENCE / "scores.json", {"niqe_stats_sha256": compute_file_sha256(stats_path),
               "rows": rows, "aggregates": aggregates, "paired_id_psnr_ci95": ci.tolist() if ci is not None else None})

    summary = ["### LOL-v2 Real official test · 100 paired captures", "",
               "The same app configuration restored every image at native size. Predictions were saved and hashed before downloading the references.", "",
               "| Method | RGB PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Shadow MAD ↓ | Paired edge correlation ↑ |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        vals = [aggregates[method][m]["mean"] for m in ("psnr_db", "ssim", "lpips_alex", "niqe", "shadow_mad", "edge_corr")]
        cells = " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals)
        summary.append(f"| {method.capitalize()} | {cells} |")
    if ci is not None:
        summary += ["", f"Hybrid minus deep PSNR: {diffs.mean():+.2f} dB. Paired bootstrap 95% interval [{ci[0]:+.2f}, {ci[1]:+.2f}] dB, resampled by pair ID because scene grouping is unavailable."]
    summary += ["", f"CPU decode + three outputs + map rendering + PNG writes: {np.median(times)/1000:.2f} s median, "
                f"{np.percentile(times, 95)/1000:.2f} s p95, {max(times)/1000:.2f} s maximum. Browser transfer is timed separately.",
                "", "This is an audit of the shipped settings. Five test inputs previously appeared in the checkpoint handoff without references. "
                "The test split was not used to change the app's inference settings during this audit."]
    (EVIDENCE / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    failures = []
    for criterion, ordered in (
        ("lowest hybrid PSNR", sorted(hybrid, key=lambda r: float(r["psnr_db"]))),
        ("largest hybrid PSNR drop versus deep", sorted(hybrid, key=lambda r: float(r["psnr_db"]) - float(deep[r["pair_id"]]["psnr_db"]))),
        ("largest fraction of newly near-clipped pixels", sorted(hybrid, key=lambda r: r["new_clipping_pct"], reverse=True)),
    ):
        selected = next(r for r in ordered if r["pair_id"] not in {c["pair_id"] for c in failures})
        pair_id = selected["pair_id"]
        images = [load_oriented_rgb_uint8(folder / f"{pair_id}.png")[0] for folder in
                  (DATA / "Input", DATA / "GT", *(RUN / m for m in METHODS))]
        error = np.mean(np.abs(images[4].astype(np.float32) - images[1]), axis=2)
        _, _, _, (x, y) = cv2.minMaxLoc(cv2.boxFilter(error, -1, (128, 128)))
        left, top = np.clip(x - 64, 0, images[0].shape[1] - 128), np.clip(y - 64, 0, images[0].shape[0] - 128)
        crop = (int(left), int(top), int(left + 128), int(top + 128))
        grid = Image.new("RGB", (5 * 128, 148))
        draw = ImageDraw.Draw(grid)
        for i, (name, image) in enumerate(zip(("Input", "Reference", *METHODS), images, strict=True)):
            draw.text((128 * i + 5, 3), name, fill="white")
            grid.paste(Image.fromarray(image).crop(crop), (128 * i, 20))
        grid.save(RUN / f"failure_{pair_id}.png")
        failures.append({"criterion": criterion, "pair_id": pair_id, "crop": crop,
                         "hybrid_psnr": selected["psnr_db"], "deep_psnr": deep[pair_id]["psnr_db"],
                         "new_clipping_pct": selected["new_clipping_pct"]})
    write_json(EVIDENCE / "failure_cases.json", failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "score"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    (infer if args.phase == "infer" else score)(test_ids())
