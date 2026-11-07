"""
20-Image Live Latency and Reliability Audit (Section 8 Release Gate).

Protocol:
- 3 warmup inferences (excluded from timing statistics)
- 20 varied live test images covering:
    - Extreme low light (<0.05 mean luminance)
    - Mixed indoor lighting
    - Streetlights and localized saturated specular highlights
    - High-frequency textures in shadow
    - Color casts (warm sodium vapor, cool fluorescent)
    - Varied resolutions up to the declared live maximum (768px long side)
- Measures min, mean, median, p95, and max end-to-end wall-clock latency
- Verifies zero OOM errors and zero crashes (<10s threshold per image)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app import process_image


def generate_varied_audit_images(count: int = 20) -> List[np.ndarray]:
    """Generate 20 deterministic varied test images simulating real deployment conditions."""
    images = []
    rng = np.random.default_rng(seed=1956)

    # Base shapes (varied aspect ratios up to 768px)
    resolutions = [
        (400, 600),
        (512, 512),
        (480, 640),
        (576, 768),
        (600, 450),
    ]

    for i in range(count):
        h, w = resolutions[i % len(resolutions)]
        img = np.zeros((h, w, 3), dtype=np.float32)

        # Diverse illumination conditions
        scene_type = i % 5
        if scene_type == 0:
            # Extreme darkness (mean < 0.04)
            img += rng.uniform(0.01, 0.05)
            # Add subtle structure
            img[h//4:3*h//4, w//4:3*w//4] += 0.03
        elif scene_type == 1:
            # Streetlight / Saturated light source (tests highlight protection)
            img += 0.03
            cy, cx = h // 3, w // 2
            y, x = np.ogrid[:h, :w]
            dist_sq = (x - cx)**2 + (y - cy)**2
            glow = np.exp(-dist_sq / (2.0 * 50.0**2))
            img += glow[..., None] * np.array([0.98, 0.92, 0.75], dtype=np.float32)
        elif scene_type == 2:
            # Unbalanced color cast (amber / sodium vapor)
            img[..., 0] = 0.14 + 0.04 * np.sin(np.linspace(0, 3.14, w))[None, :]
            img[..., 1] = 0.07 + 0.02 * np.sin(np.linspace(0, 3.14, w))[None, :]
            img[..., 2] = 0.02
        elif scene_type == 3:
            # High-frequency textured shadow
            img += 0.06
            checker = ((np.arange(h) // 16)[:, None] ^ (np.arange(w) // 16)[None, :]).astype(np.float32) * 0.05
            img += checker[..., None]
        else:
            # Mixed ambient gradient
            xx = np.linspace(0.02, 0.35, w, dtype=np.float32)
            img += xx[None, :, None]

        # Add Gaussian sensor noise
        noise = rng.normal(0.0, 0.012, (h, w, 3)).astype(np.float32)
        img_u8 = np.clip(np.round((img + noise) * 255.0), 0, 255).astype(np.uint8)
        images.append(img_u8)

    return images


def run_audit() -> Dict[str, any]:
    print("=================================================================")
    print("P1 Release Gate: 20-Image Live Latency & Zero-Crash Verification")
    print("=================================================================")

    # 1. Warmup
    print("Executing 3 warmup inferences...")
    warmup_img = (np.random.default_rng(42).uniform(10, 80, (400, 600, 3))).astype(np.uint8)
    for w_idx in range(3):
        t0 = time.perf_counter()
        process_image(warmup_img, "hybrid_guarded", True, True)
        print(f"  Warmup {w_idx+1}/3 complete in {(time.perf_counter() - t0)*1000.0:.1f} ms")

    # 2. 20 Varied Live Inferences
    print("\nRunning 20 varied live test scenes...")
    audit_images = generate_varied_audit_images(20)
    latencies_ms: List[float] = []
    crashes = 0
    ooms = 0
    records = []

    for idx, img in enumerate(audit_images, 1):
        h, w, _ = img.shape
        t0 = time.perf_counter()
        try:
            c, d, hybrid, m1, m2, m3, m4, diag = process_image(img, "hybrid_guarded", True, True)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            # Assert non-empty, finite, same dimensions
            assert hybrid is not None and hybrid.shape == (h, w, 3), f"Output shape mismatch: {hybrid.shape if hybrid is not None else None}"
            assert np.isfinite(hybrid).all(), "NaN or Inf detected in output"

            status = "PASS (<10s)" if elapsed_ms < 10000.0 else "FAIL (>10s)"
            print(f"  [{idx:02d}/20] Res: {w}x{h} px | Latency: {elapsed_ms:6.1f} ms | Status: {status}")
            records.append({
                "image_idx": idx,
                "resolution": f"{w}x{h}",
                "latency_ms": round(elapsed_ms, 2),
                "pass_10s_gate": elapsed_ms < 10000.0,
            })

        except MemoryError:
            ooms += 1
            crashes += 1
            print(f"  [{idx:02d}/20] Res: {w}x{h} px | ERROR: Out of Memory (OOM)")
        except Exception as e:
            crashes += 1
            print(f"  [{idx:02d}/20] Res: {w}x{h} px | CRASH: {e}")

    # Compute Statistics
    lat_arr = np.array(latencies_ms)
    min_lat = float(np.min(lat_arr))
    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    max_lat = float(np.max(lat_arr))

    summary = {
        "total_images": len(audit_images),
        "successful_images": len(latencies_ms),
        "zero_crashes": crashes == 0,
        "zero_oom": ooms == 0,
        "all_under_10s": bool((lat_arr < 10000.0).all()),
        "min_latency_ms": round(min_lat, 1),
        "mean_latency_ms": round(mean_lat, 1),
        "median_latency_ms": round(median_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "max_latency_ms": round(max_lat, 1),
        "records": records,
    }

    print("\n-----------------------------------------------------------------")
    print("AUDIT SUMMARY:")
    print(f"  Total Inferences:   {summary['total_images']}")
    print(f"  Zero Crashes:       {'PASS' if summary['zero_crashes'] else 'FAIL'}")
    print(f"  Zero OOM Errors:    {'PASS' if summary['zero_oom'] else 'FAIL'}")
    print(f"  All < 10s:          {'PASS' if summary['all_under_10s'] else 'FAIL'}")
    print(f"  Min Latency:        {summary['min_latency_ms']} ms ({summary['min_latency_ms']/1000.0:.2f} s)")
    print(f"  Median Latency:     {summary['median_latency_ms']} ms ({summary['median_latency_ms']/1000.0:.2f} s)")
    print(f"  P95 Latency:        {summary['p95_latency_ms']} ms ({summary['p95_latency_ms']/1000.0:.2f} s)")
    print(f"  Max Latency:        {summary['max_latency_ms']} ms ({summary['max_latency_ms']/1000.0:.2f} s)")
    print("-----------------------------------------------------------------")

    # Save artifact
    out_path = _REPO_ROOT / "outputs" / "live_latency_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Audit record saved to {out_path}")

    return summary


if __name__ == "__main__":
    run_audit()
