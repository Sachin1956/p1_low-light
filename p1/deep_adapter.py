"""
Part B: Deep model adapter for HVI-CIDNet.

Strictly follows docs/P1_ML_EXECUTION_PLAN.md & docs/P1_AGENT_CONTRACT.md:
- No GT mean scaling (--use_GT_mean disabled / removed from inference).
- Full input-only inference; no references passed.
- Factor-8 reflection padding to handle arbitrary dimensions without resizing.
- Strict=True weight loading with checkpoint SHA-256 verification.
- Deterministic eval mode.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure external/HVI-CIDNet is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTERNAL_CIDNET = _REPO_ROOT / "external" / "HVI-CIDNet"
if str(_EXTERNAL_CIDNET) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_CIDNET))

from net.CIDNet import CIDNet  # type: ignore

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[CIDNet, str]] = {}


def load_cidnet_model(checkpoint_path: str | Path, device: str = "cpu") -> Tuple[CIDNet, str]:
    """
    Load CIDNet model with strict state_dict loading and SHA256 provenance hash.
    Caches model in memory so repeat inferences do not reload from disk.
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_absolute():
        ckpt_p = _REPO_ROOT / ckpt_p

    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    ckpt_bytes = ckpt_p.read_bytes()
    ckpt_hash = hashlib.sha256(ckpt_bytes).hexdigest()

    cache_key = (str(ckpt_p.resolve()), device)
    if cache_key in _MODEL_CACHE:
        cached_model, cached_hash = _MODEL_CACHE[cache_key]
        if cached_hash == ckpt_hash:
            return cached_model, cached_hash

    # Build model architecture
    model = CIDNet()
    # Explicitly disable any test-finetuned GT multipliers
    model.trans.gated = False
    model.trans.gated2 = False
    model.trans.alpha = 1.0

    state_dict = torch.load(ckpt_p, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Clean potential DataParallel 'module.' prefix if present
    cleaned_sd = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_sd[key] = v

    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, ckpt_hash)
    return model, ckpt_hash


def run_deep_inference(
    rgb_uint8: np.ndarray,
    checkpoint_path: str | Path,
    device: str = "auto",
    use_amp: bool = False,
    gamma_input: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any], float, str]:
    """
    Inference pass for HVI-CIDNet on a single image.
    Returns:
      - restored_uint8 (HxWx3 uint8)
      - params dict
      - elapsed_ms float
      - checkpoint_hash str
    """
    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    model, ckpt_hash = load_cidnet_model(checkpoint_path, device=target_device)

    h, w, _ = rgb_uint8.shape
    factor = 8
    pad_h = (factor - (h % factor)) % factor
    pad_w = (factor - (w % factor)) % factor

    # Convert to float tensor [0, 1] in shape (1, 3, H, W)
    x = torch.from_numpy(rgb_uint8.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    if pad_h > 0 or pad_w > 0:
        # Pad right and bottom using reflect padding
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        x_padded = x

    x_padded = x_padded.to(target_device)
    if gamma_input != 1.0:
        x_padded = torch.clamp(x_padded, 1e-6, 1.0).pow(gamma_input)

    if target_device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        if use_amp and target_device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x_padded)
        else:
            out = model(x_padded)

        out = torch.clamp(out, 0.0, 1.0)
        # Crop back to original dimensions
        out = out[:, :, :h, :w]

    if target_device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out_uint8 = np.clip(np.round(out_np * 255.0), 0, 255).astype(np.uint8)

    params = {
        "model_architecture": "HVI-CIDNet",
        "checkpoint": str(checkpoint_path),
        "device": target_device,
        "use_amp": use_amp,
        "gamma_input": gamma_input,
        "input_shape": [h, w, 3],
        "pad": [pad_h, pad_w],
        "gt_mean_correction": False,  # Explicitly verified false
    }

    return out_uint8, params, elapsed_ms, ckpt_hash
