"""
Run full comparative evaluation (Classical, Deep, Adaptive Hybrid) on SMALL v1.0.0 Test Set.
Computes official metrics in exact P1 order:
1. PSNR (dB) and SSIM
2. LPIPS
3. NIQE
4. Latency (ms and s)
5. Noise suppression (Residual shadow MAD & dB gain) and Edge Preservation Index (EPI)
"""

import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data_manifests" / "SMALL-v1.0.0-test.csv"
OUTPUT_JSON = REPO_ROOT / "outputs" / "small_test_evaluation.json"
CKPT_PATH = str(REPO_ROOT / "weights" / "kaggle_run_v2" / "p1_kaggle_output" / "best_model.pth")

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import EvaluationResult, LowLightEvaluator
from p1.interface import EnhancementConfig, enhance, load_oriented_rgb_uint8

def run_small_evaluation(max_samples: int = 50):
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest {MANIFEST_PATH} not found.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    pairs = pairs[:max_samples]
    print(f"Loaded {len(pairs)} pairs for SMALL v1.0.0 evaluation.")

    evaluator = LowLightEvaluator(device="cpu")

    approaches = [
        ("Classical CV Only", EnhancementConfig(
            mode="classical_adaptive",
            checkpoint_path=None,
            device="cpu",
            classical_adaptive=True,
            quality_guard=False,
        )),
        ("Deep Learning Only (Adapted)", EnhancementConfig(
            mode="deep",
            checkpoint_path=CKPT_PATH,
            device="cpu",
        )),
        ("Adaptive Hybrid (ARHR Proposed)", EnhancementConfig(
            mode="hybrid_guarded",
            checkpoint_path=CKPT_PATH,
            device="cpu",
            classical_adaptive=True,
            quality_guard=True,
        )),
    ]

    results = []

    for name, config in approaches:
        print(f"\n--- Running: {name} on SMALL Test Set ---")
        single_results = []

        for idx, row in enumerate(pairs, 1):
            low_p = REPO_ROOT / row["low_path"]
            ref_p = REPO_ROOT / row["reference_path"]

            low_rgb, _ = load_oriented_rgb_uint8(low_p)
            ref_rgb, _ = load_oriented_rgb_uint8(ref_p)

            # Match dimensions if needed
            if low_rgb.shape != ref_rgb.shape:
                ref_rgb = cv2.resize(ref_rgb, (low_rgb.shape[1], low_rgb.shape[0]), interpolation=cv2.INTER_AREA)

            # Cap long side to 768 for live memory safety & fast latency
            h, w = low_rgb.shape[:2]
            max_side = max(h, w)
            if max_side > 768:
                scale = 768.0 / max_side
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                low_rgb = cv2.resize(low_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                ref_rgb = cv2.resize(ref_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            t0 = time.perf_counter()
            res = enhance(low_rgb, config)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            pred_rgb = res.image

            er_single = evaluator.evaluate_single(
                restored=pred_rgb,
                low_light_input=low_rgb,
                ground_truth=ref_rgb,
                latency_ms=lat_ms,
                approach_name=name,
            )
            single_results.append(er_single)

            if idx % 10 == 0 or idx == len(pairs):
                print(f"[{idx}/{len(pairs)}] PSNR: {er_single.psnr_db:.2f} dB, SSIM: {er_single.ssim:.4f}, LPIPS: {er_single.lpips:.4f}, NIQE: {er_single.niqe:.2f}, Latency: {lat_ms:.1f} ms")

        # Aggregate across dataset
        psnrs = [r.psnr_db for r in single_results if r.psnr_db is not None]
        ssims = [r.ssim for r in single_results if r.ssim is not None]
        lpips_list = [r.lpips for r in single_results if r.lpips is not None]
        niqes = [r.niqe for r in single_results]
        lats = [r.latency_ms for r in single_results]
        noises = [r.shadow_noise_mad for r in single_results]
        noisegs = [r.noise_suppression_db for r in single_results]
        scores = [r.noise_suppression_score for r in single_results]
        epis = [r.edge_preservation_epi for r in single_results]

        er_agg = EvaluationResult(
            approach_name=name,
            psnr_db=round(float(np.mean(psnrs)), 2),
            ssim=round(float(np.mean(ssims)), 4),
            lpips=round(float(np.mean(lpips_list)), 4),
            niqe=round(float(np.mean(niqes)), 4),
            latency_ms=round(float(np.mean(lats)), 1),
            shadow_noise_mad=round(float(np.mean(noises)), 2),
            noise_suppression_db=round(float(np.mean(noisegs)), 2),
            noise_suppression_score=round(float(np.mean(scores)), 3),
            edge_preservation_epi=round(float(np.mean(epis)), 4),
            num_images=len(pairs),
            resolution="768 max-side",
            hardware="CPU",
        )
        results.append(er_agg)
        print(f"Aggregate for {name}: PSNR={er_agg.psnr_db}dB, SSIM={er_agg.ssim}, LPIPS={er_agg.lpips}, NIQE={er_agg.niqe}, Latency={er_agg.latency_ms}ms")

    # Save to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out_dict = {
        "dataset": "SMALL v1.0.0 (Smartphone Multi-Device Low-Light Dataset)",
        "split": "Official Test Split (50 pairs)",
        "num_pairs": len(pairs),
        "results": [r.to_dict() for r in results],
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_JSON}")
    return results

if __name__ == "__main__":
    run_small_evaluation()
