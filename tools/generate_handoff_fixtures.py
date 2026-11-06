"""
Generate the five canonical P1 handoff fixtures defined in docs/P1_ML_EXECUTION_PLAN.md:
1. dark_indoor: Uniform low ambient light with subtle surface gradients and shadow noise.
2. streetlight: Dark outdoor scene with a localized saturated light source (tests highlight protection).
3. color_cast: Low-light scene with an unbalanced warm/cool color cast (tests color fidelity).
4. textured_shadow: Dark scene with fine geometric textures and sensor noise (tests edge preservation).
5. already_bright_region: High-key scene with bright areas and mild shadows (tests gain restraint).

Saves lossless PNG fixtures to `data/handoff_fixtures/` and runs Part A classical inference
saving predictions, diagnostic maps, and JSONL provenance records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import classical
from classical.config import ClassicalConfig, PipelineMode


def generate_canonical_fixtures(output_dir: Path) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=1956)
    fixtures = []

    # 1. Dark Indoor (400x600)
    img1 = np.zeros((400, 600, 3), dtype=np.float32)
    # Ambient illumination gradient from left to right [0.03 to 0.12]
    xx = np.linspace(0.03, 0.12, 600, dtype=np.float32)
    img1 += xx[np.newaxis, :, np.newaxis]
    # Add room elements (desk/furniture shapes)
    img1[220:380, 50:450, 0] += 0.05
    img1[220:380, 50:450, 1] += 0.03
    img1[220:380, 50:450, 2] += 0.02
    # Sensor noise in shadows
    noise1 = rng.normal(0.0, 0.015, (400, 600, 3)).astype(np.float32)
    img1_u8 = np.clip((img1 + noise1) * 255.0, 0, 255).astype(np.uint8)
    p1 = output_dir / "handoff_1_dark_indoor.png"
    cv2.imwrite(str(p1), cv2.cvtColor(img1_u8, cv2.COLOR_RGB2BGR))
    fixtures.append({"image_id": "handoff_1_dark_indoor", "input_path": str(p1), "description": "Dark indoor with shadow noise"})

    # 2. Streetlight (400x600)
    img2 = np.zeros((400, 600, 3), dtype=np.float32) + 0.02
    # Add road and buildings
    img2[300:400, :, :] += 0.04
    # Saturated streetlight at (150, 300) with radial glow
    y, x = np.ogrid[:400, :600]
    dist_sq = (x - 300)**2 + (y - 150)**2
    glow = np.exp(-dist_sq / (2.0 * 45.0**2)).astype(np.float32)
    lamp_core = (dist_sq < 18**2).astype(np.float32)
    img2 += glow[..., np.newaxis] * np.array([0.95, 0.90, 0.70], dtype=np.float32)
    img2 += lamp_core[..., np.newaxis] * 0.99
    noise2 = rng.normal(0.0, 0.012, (400, 600, 3)).astype(np.float32)
    img2_u8 = np.clip((img2 + noise2) * 255.0, 0, 255).astype(np.uint8)
    p2 = output_dir / "handoff_2_streetlight.png"
    cv2.imwrite(str(p2), cv2.cvtColor(img2_u8, cv2.COLOR_RGB2BGR))
    fixtures.append({"image_id": "handoff_2_streetlight", "input_path": str(p2), "description": "Streetlight with strong saturated highlight and radial halo"})

    # 3. Color Cast (400x600)
    img3 = np.zeros((400, 600, 3), dtype=np.float32)
    # Heavy warm sodium-vapor amber tint (R > G >> B)
    img3[..., 0] = 0.15 + 0.05 * np.sin(np.linspace(0, 3.14, 600))[np.newaxis, :]
    img3[..., 1] = 0.08 + 0.03 * np.sin(np.linspace(0, 3.14, 600))[np.newaxis, :]
    img3[..., 2] = 0.02
    noise3 = rng.normal(0.0, 0.010, (400, 600, 3)).astype(np.float32)
    img3_u8 = np.clip((img3 + noise3) * 255.0, 0, 255).astype(np.uint8)
    p3 = output_dir / "handoff_3_color_cast.png"
    cv2.imwrite(str(p3), cv2.cvtColor(img3_u8, cv2.COLOR_RGB2BGR))
    fixtures.append({"image_id": "handoff_3_color_cast", "input_path": str(p3), "description": "Low light with strong amber/sodium color cast"})

    # 4. Textured Shadow (400x600)
    img4 = np.zeros((400, 600, 3), dtype=np.float32) + 0.05
    # Checkerboard/brick texture in dark shadow
    grid_x = (np.arange(600) // 16) % 2
    grid_y = (np.arange(400) // 16) % 2
    checker = (grid_y[:, np.newaxis] ^ grid_x[np.newaxis, :]).astype(np.float32) * 0.04
    img4[80:320, 100:500, :] += checker[80:320, 100:500, np.newaxis]
    noise4 = rng.normal(0.0, 0.018, (400, 600, 3)).astype(np.float32)
    img4_u8 = np.clip((img4 + noise4) * 255.0, 0, 255).astype(np.uint8)
    p4 = output_dir / "handoff_4_textured_shadow.png"
    cv2.imwrite(str(p4), cv2.cvtColor(img4_u8, cv2.COLOR_RGB2BGR))
    fixtures.append({"image_id": "handoff_4_textured_shadow", "input_path": str(p4), "description": "Fine geometric texture inside noisy shadow"})

    # 5. Already Bright Region (400x600)
    img5 = np.zeros((400, 600, 3), dtype=np.float32) + 0.65
    # Naturally bright sky / lit room with mild shadowed corner
    img5[280:400, 0:250, :] = 0.20
    noise5 = rng.normal(0.0, 0.005, (400, 600, 3)).astype(np.float32)
    img5_u8 = np.clip((img5 + noise5) * 255.0, 0, 255).astype(np.uint8)
    p5 = output_dir / "handoff_5_already_bright.png"
    cv2.imwrite(str(p5), cv2.cvtColor(img5_u8, cv2.COLOR_RGB2BGR))
    fixtures.append({"image_id": "handoff_5_already_bright", "input_path": str(p5), "description": "Predominantly bright scene with mild corner shadow"})

    return fixtures


def main() -> None:
    fixtures_dir = Path("data/handoff_fixtures")
    manifest = generate_canonical_fixtures(fixtures_dir)
    print(f"Generated {len(manifest)} canonical handoff fixtures in {fixtures_dir}")

    # Run E0 Baseline
    cfg_e0 = ClassicalConfig(mode=PipelineMode.E0_BASELINE)
    log_e0 = classical.run_batch_inference(manifest, "outputs/classical/E0_baseline", cfg_e0)
    print(f"E0 inference complete -> {log_e0}")

    # Run E1 Fixed Classical
    cfg_e1 = ClassicalConfig(mode=PipelineMode.E1_FIXED)
    log_e1 = classical.run_batch_inference(manifest, "outputs/classical/E1_fixed", cfg_e1)
    print(f"E1 inference complete -> {log_e1}")

    # Run E2 Adaptive Classical
    cfg_e2 = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)
    log_e2 = classical.run_batch_inference(manifest, "outputs/classical/E2_adaptive", cfg_e2)
    print(f"E2 inference complete -> {log_e2}")


if __name__ == "__main__":
    main()
