"""
Report generator for Problem Statement 1 (P1):
Generates the mandatory `eval_P1_T<TeamID>.md` file strictly following the
Computer Vision Hackathon 2026 Instruction Sheet (Page 15):

1. First line MUST be all team members' USNs separated by commas.
2. Metrics MUST appear in the exact order listed in Problem Statement 1 (Page 2):
   - Metric 1: PSNR and SSIM (when ground-truth references are available)
   - Metric 2: LPIPS for perceptual quality
   - Metric 3: NIQE (no-reference quality metric)
   - Metric 4: Processing latency
   - Metric 5: Noise suppression and edge preservation
3. Includes comparative evaluation across Classical Only, Deep Learning Only,
   and Hybrid approaches, plus progressive Ablation Study and Failure-Case Analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .metrics import EvaluationResult


def _fmt_opt(val: Optional[float], decimals: int = 4, suffix: str = "") -> str:
    if val is None:
        return "N/A (Unpaired)"
    return f"{val:.{decimals}f}{suffix}"


def format_comparative_markdown(
    results: Sequence[EvaluationResult],
    dataset_name: str = "LOL-v2 Validation / Test Split",
) -> str:
    """
    Format a comparative Markdown table with columns in the exact order of
    Problem Statement 1 (Page 2 of Computer Vision Hackathon 2026 PDF).
    """
    lines = [
        f"### Comparative Evaluation — {dataset_name}",
        "",
        "| Approach | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Processing Latency (ms / s) ↓ | 5a. Noise Suppression (Residual $\\hat{\\sigma}_{\\text{MAD}}$ ↓ / Gain dB ↑) | 5b. Edge Preservation (EPI ↑) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        psnr_s = _fmt_opt(r.psnr_db, 2)
        ssim_s = _fmt_opt(r.ssim, 4)
        lpips_s = _fmt_opt(r.lpips, 4)
        niqe_s = f"{r.niqe:.4f}"
        lat_s = f"{r.latency_ms:.1f} ms ({r.latency_sec:.3f} s)"
        noise_s = f"{r.shadow_noise_mad:.2f} / +{r.noise_suppression_db:.2f} dB (score {r.noise_suppression_score:.3f})"
        epi_s = f"{r.edge_preservation_epi:.4f}"
        lines.append(
            f"| **{r.approach_name}** | {psnr_s} | {ssim_s} | {lpips_s} | {niqe_s} | {lat_s} | {noise_s} | {epi_s} |"
        )
    return "\n".join(lines)


def format_ablation_markdown(
    ablation_results: Sequence[EvaluationResult],
) -> str:
    """
    Format the progressive ablation study table (Experiments A through F)
    showing the contribution of each pipeline component.
    """
    lines = [
        "### Ablation Study — Component Contribution Analysis",
        "",
        "| Experiment / Configuration | 1. PSNR (dB) ↑ | 1. SSIM ↑ | 2. LPIPS ↓ | 3. NIQE ↓ | 4. Latency (ms) ↓ | 5a. Shadow Noise $\\hat{\\sigma}_{\\text{MAD}}$ ↓ | 5b. Edge Preservation (EPI ↑) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ablation_results:
        psnr_s = _fmt_opt(r.psnr_db, 2)
        ssim_s = _fmt_opt(r.ssim, 4)
        lpips_s = _fmt_opt(r.lpips, 4)
        lines.append(
            f"| {r.approach_name} | {psnr_s} | {ssim_s} | {lpips_s} | {r.niqe:.4f} | {r.latency_ms:.1f} | {r.shadow_noise_mad:.2f} | {r.edge_preservation_epi:.4f} |"
        )
    return "\n".join(lines)


def generate_eval_markdown_file(
    output_path: str | Path,
    usns: Sequence[str],
    comparative_results: Sequence[EvaluationResult],
    ablation_results: Optional[Sequence[EvaluationResult]] = None,
    unseen_results: Optional[Sequence[EvaluationResult]] = None,
    dataset_name: str = "LOL-v2 Real (Validation / Official Test Split)",
    team_id: str = "T01",
    failure_cases: Optional[List[Dict[str, str]]] = None,
    json_export_path: Optional[str | Path] = None,
) -> Path:
    """
    Write the mandatory `eval_P1_<TeamID>.md` deliverable file.

    Strictly enforces:
      - Line 1: Comma-separated USNs ONLY (`USN1, USN2, USN3, USN4`)
      - Exact P1 metric ordering (1. PSNR & SSIM, 2. LPIPS, 3. NIQE, 4. Processing Latency,
        5. Noise Suppression & Edge Preservation)
    """
    cleaned_usns = [u.strip() for u in usns if u and u.strip()]
    if not cleaned_usns:
        cleaned_usns = [
            "1RUA24CSE0375",
            "1RUA24CSE0376",
            "1RUA24CSE0384",
            "1RUA24CSE0395",
        ]
    first_line = ", ".join(cleaned_usns)

    hw = comparative_results[0].hardware if comparative_results else "CPU"
    res = comparative_results[0].resolution if comparative_results else "600x400"
    n_imgs = comparative_results[0].num_images if comparative_results else 1

    # Individual metric breakdown in exact document order
    lines: List[str] = [
        first_line,
        "",
        f"# Comparative Evaluation Results — Problem Statement 1 (`eval_P1_{team_id}`)",
        "",
        "## Team Members",
        "| Name | USN |",
        "|---|---|",
        "| Rohit Ajit Bharadwaj | `1RUA24CSE0375` |",
        "| Rohith P Hegde | `1RUA24CSE0376` |",
        "| S Sachithananthan | `1RUA24CSE0384` |",
        "| Saket M | `1RUA24CSE0395` |",
        "",
        "## Evaluation Protocol & Environment",
        f"- **Problem ID:** P1 — Robust Low-Light Image Restoration and Enhancement",
        f"- **Dataset & Split:** {dataset_name} ({n_imgs} images evaluated)",
        f"- **Input Resolution:** {res}",
        f"- **Hardware:** {hw}",
        "- **Timing Protocol:** End-to-end inference time per image (including image characteristic analysis, restoration, spatial fusion, and quality-aware refinement).",
        "",
        "---",
        "",
        "## 1. Comparative Evaluation Summary (Exact P1 Metric Order)",
        "",
        format_comparative_markdown(comparative_results, dataset_name=dataset_name),
        "",
        "---",
        "",
        "## 2. Detailed Metric Breakdown (In Official Document Order)",
        "",
        "### Metric 1: PSNR and SSIM (Full-Reference Reconstruction & Structural Fidelity)",
        "- **Definition:** Peak Signal-to-Noise Ratio (dB, higher is better) and Structural Similarity Index ($[0, 1]$, higher is better) computed against paired normal-light ground truth.",
        "",
        "| Approach | PSNR (dB) ↑ | SSIM ↑ |",
        "|---|---:|---:|",
    ]
    for r in comparative_results:
        lines.append(
            f"| {r.approach_name} | {_fmt_opt(r.psnr_db, 2)} | {_fmt_opt(r.ssim, 4)} |"
        )

    lines.extend(
        [
            "",
            "### Metric 2: LPIPS (Learned Perceptual Image Patch Similarity)",
            "- **Definition:** Deep feature perceptual distance (AlexNet backbone, lower is better) measuring human-aligned texture and perceptual fidelity.",
            "",
            "| Approach | LPIPS ↓ |",
            "|---|---:|",
        ]
    )
    for r in comparative_results:
        lines.append(f"| {r.approach_name} | {_fmt_opt(r.lpips, 4)} |")

    lines.extend(
        [
            "",
            "### Metric 3: NIQE (No-Reference Image Quality Metric)",
            "- **Definition:** Natural Image Quality Evaluator based on Mean Subtracted Contrast Normalized (MSCN) Natural Scene Statistics (lower is better).",
            "",
            "| Approach | NIQE ↓ |",
            "|---|---:|",
        ]
    )
    for r in comparative_results:
        lines.append(f"| {r.approach_name} | {r.niqe:.4f} |")

    lines.extend(
        [
            "",
            "### Metric 4: Processing Latency",
            "- **Definition:** Mean end-to-end processing time per image in milliseconds and seconds (lower is better; must remain well under the $< 10\\text{ s}$ live deployment requirement).",
            "",
            "| Approach | Mean Latency (ms) ↓ | Mean Latency (s) ↓ | Meets $< 10\\text{ s}$ Rubric Target |",
            "|---|---:|---:|:---:|",
        ]
    )
    for r in comparative_results:
        ok = "Yes" if r.latency_sec < 10.0 else "No"
        lines.append(
            f"| {r.approach_name} | {r.latency_ms:.2f} ms | {r.latency_sec:.3f} s | {ok} |"
        )

    lines.extend(
        [
            "",
            "### Metric 5: Noise Suppression and Edge Preservation",
            "- **Noise Suppression Definition:** High-frequency diagonal wavelet Median Absolute Deviation ($\\hat{\\sigma}_{\\text{MAD}} = \\text{median}(|W_{HH}|)/0.6745$) measured in homogeneous shadow regions (lower residual $\\hat{\\sigma}_{\\text{MAD}}$ and higher gain-normalized suppression dB are better).",
            "- **Edge Preservation Definition:** Edge Preservation Index (EPI $\\in [0, 1]$, higher is better) combining Sobel gradient magnitude correlation and FSIM-style boundary similarity.",
            "",
            "| Approach | Residual Shadow Noise $\\hat{\\sigma}_{\\text{MAD}}$ ↓ | Gain-Normalized Noise Suppression (dB) ↑ | Noise Cleanliness Score $[0,1]$ ↑ | Edge Preservation Index (EPI) ↑ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for r in comparative_results:
        lines.append(
            f"| {r.approach_name} | {r.shadow_noise_mad:.2f} | +{r.noise_suppression_db:.2f} dB | {r.noise_suppression_score:.3f} | {r.edge_preservation_epi:.4f} |"
        )

    if unseen_results:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Unseen Low-Light Generalization Evaluation (No-Reference)",
                "",
                format_comparative_markdown(
                    unseen_results,
                    dataset_name="Unseen Real-World Low-Light Test Images (Unpaired)",
                ),
            ]
        )

    if ablation_results:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 4. Ablation Study",
                "",
                format_ablation_markdown(ablation_results),
            ]
        )

    default_failures = failure_cases or [
        {
            "case": "Case 1 — Crushed Blacks / Zero-Bit Quantization Shadows",
            "condition": "Extremely dark regions where raw RGB intensities are quantized to (0, 0, 0).",
            "symptom": "Loss of fine micro-texture and posterization in deepest shadow pockets.",
            "cause": "When photon counts fall below the sensor read-noise floor and quantize to 0, true spatial signal is physically absent; non-hallucinatory restoration avoids inventing synthetic textures.",
        },
        {
            "case": "Case 2 — Severe Mixed-Illuminant Scenes",
            "condition": "Simultaneous warm sodium-vapor streetlights (2200K) and cool fluorescent/LED signage (6000K).",
            "symptom": "Residual warm or cool tint in localized transition zones.",
            "cause": "Global and smooth spatial white-balance priors assume slowly varying illumination chromaticity and can under-correct sharp dual-illuminant boundaries.",
        },
        {
            "case": "Case 3 — Low-Light Motion Blur & Defocus",
            "condition": "Handheld long-exposure capture with camera shake or moving subjects.",
            "symptom": "Restored brightness reveals motion-blurred edges cleanly without deblurring them.",
            "cause": "Retinex decomposition and HVI-CIDNet restore photometric illumination and suppress sensor noise, but do not perform blind motion point-spread-function (PSF) deconvolution.",
        },
        {
            "case": "Case 4 — Extreme High-ISO Sensor Noise on Fine Foliage/Fabric",
            "condition": "ISO > 12800 capture where chroma/luma noise amplitude matches fine texture gradients.",
            "symptom": "Trade-off between slight residual grain or mild softening of stochastic micro-textures.",
            "cause": "When noise variance overlaps the spatial frequency band of fine stochastic textures, edge-preserving filters face an information-theoretic limit.",
        },
    ]

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Documented Failure-Case Analysis",
            "",
            "| Failure Case | Input Condition | Observed Behavior | Root Technical Cause |",
            "|---|---|---|---|",
        ]
    )
    for fc in default_failures:
        lines.append(
            f"| **{fc['case']}** | {fc['condition']} | {fc['symptom']} | {fc['cause']} |"
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_export_path is not None:
        json_path = Path(json_export_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "usns": cleaned_usns,
            "team_id": team_id,
            "dataset": dataset_name,
            "comparative_results": [r.to_dict() for r in comparative_results],
            "ablation_results": [r.to_dict() for r in (ablation_results or [])],
            "unseen_results": [r.to_dict() for r in (unseen_results or [])],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return out_path
