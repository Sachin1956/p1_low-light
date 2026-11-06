"""
Quantitative Comparison of Deep Learning Checkpoints and Hybrid Fusion.
Compares:
1. Kaggle Fine-Tuned Checkpoint (best_model.pth)
2. Official LOL-v2 Pretrained (best_PSNR.pth)
3. Official LOL-v1 Pretrained (PSNR_24.74.pth)
4. Adaptive Hybrid using each deep checkpoint
"""

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

import classical
from classical.config import ClassicalConfig, PipelineMode
import fusion
from evaluation.metrics import LowLightEvaluator, to_float_rgb
from evaluate import create_synthetic_paired_dataset
from p1.deep_adapter import run_deep_inference


def main():
    evaluator = LowLightEvaluator(device="cpu")
    dataset = create_synthetic_paired_dataset(num_images=6, height=400, width=600)
    benchmark_pairs = [(item[1], item[2], item[0]) for item in dataset]

    checkpoints = [
        ("Kaggle_FineTuned_best_model", "weights/kaggle_run_v2/p1_kaggle_output/best_model.pth"),
        ("Official_LOLv2_best_PSNR", "weights/cidnet/CIDNETweights/lolv2/best_PSNR.pth"),
        ("Official_LOLv1_PSNR_24.74", "weights/cidnet/CIDNETweights/lolv1/PSNR_24.74.pth"),
    ]

    print(f"[Deep Benchmark] Evaluating {len(checkpoints)} checkpoints across {len(benchmark_pairs)} benchmark scenes...")

    results = []
    for label, ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            print(f"Skipping {label}: file not found at {ckpt_path}")
            continue

        per_img_deep = []
        per_img_hybrid = []

        for low_f, gt_f, scene_id in benchmark_pairs:
            low_u8 = np.clip(low_f * 255.0, 0, 255).astype(np.uint8)

            # Deep inference
            t0 = time.perf_counter()
            deep_u8, _, lat_deep, _ = run_deep_inference(low_u8, ckpt_path, device="cpu")
            deep_f = to_float_rgb(deep_u8)

            res_d = evaluator.evaluate_single(
                restored=deep_f,
                ground_truth=gt_f,
                low_light_input=low_f,
                latency_ms=lat_deep,
                approach_name=f"Deep ({label})",
            )
            per_img_deep.append(res_d)

            # Classical inference (E2 Adaptive with optimal saturation)
            cfg_c = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE, saturation_factor=1.0)
            res_c = classical.enhance(low_u8, cfg_c)

            # Hybrid Fusion
            t0_h = time.perf_counter()
            fused_u8 = fusion.fuse_hybrid(low_u8, res_c.image, deep_u8)
            lat_hybrid = (time.perf_counter() - t0_h) * 1000.0 + lat_deep + res_c.elapsed_ms
            fused_f = to_float_rgb(fused_u8)

            res_h = evaluator.evaluate_single(
                restored=fused_f,
                ground_truth=gt_f,
                low_light_input=low_f,
                latency_ms=lat_hybrid,
                approach_name=f"Hybrid ({label})",
            )
            per_img_hybrid.append(res_h)

        agg_d = evaluator.aggregate_results(per_img_deep, f"Deep Only: {label}")
        agg_h = evaluator.aggregate_results(per_img_hybrid, f"Hybrid: {label}")

        print(f"\n--- Checkpoint: {label} ---")
        print(f"  Deep Only | PSNR: {agg_d.psnr_db:.2f} dB | SSIM: {agg_d.ssim:.4f} | LPIPS: {agg_d.lpips:.4f} | NIQE: {agg_d.niqe:.3f} | Latency: {agg_d.latency_ms:.1f}ms")
        print(f"  Hybrid    | PSNR: {agg_h.psnr_db:.2f} dB | SSIM: {agg_h.ssim:.4f} | LPIPS: {agg_h.lpips:.4f} | NIQE: {agg_h.niqe:.3f} | Latency: {agg_h.latency_ms:.1f}ms")

        results.append({
            "checkpoint": label,
            "deep": {
                "psnr_db": agg_d.psnr_db,
                "ssim": agg_d.ssim,
                "lpips": agg_d.lpips,
                "niqe": agg_d.niqe,
                "latency_ms": agg_d.latency_ms,
            },
            "hybrid": {
                "psnr_db": agg_h.psnr_db,
                "ssim": agg_h.ssim,
                "lpips": agg_h.lpips,
                "niqe": agg_h.niqe,
                "latency_ms": agg_h.latency_ms,
            }
        })

    out_file = Path("outputs/checkpoint_comparison.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved comparison records to: {out_file}")


if __name__ == "__main__":
    main()
