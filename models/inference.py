"""
Inference wrapper for HVI-CIDNet.
Enforces whole-image policy (pad to multiple of 16, crop padding).
Strictly adheres to Reference Isolation:
NO Ground-Truth mean scaling or reference-dependent exposure adjustment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from models.hvi_cidnet import CIDNet

_GLOBAL_MODEL: Optional[CIDNet] = None
_MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(checkpoint_path: Optional[Union[str, Path]] = None) -> CIDNet:
    """Load or initialize CIDNet in eval mode."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is not None and checkpoint_path is None:
        return _GLOBAL_MODEL

    model = CIDNet(base_channels=32)
    model.to(_MODEL_DEVICE)

    if checkpoint_path is not None and os.path.exists(str(checkpoint_path)):
        state_dict = torch.load(str(checkpoint_path), map_location=_MODEL_DEVICE, weights_only=True)
        # Handle DataParallel prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    if checkpoint_path is None:
        _GLOBAL_MODEL = model
    return model


def enhance_deep(
    rgb: np.ndarray,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Run feedforward deep inference on input image.

    Inputs:
        rgb: HxWx3 uint8 [0, 255] or float32 [0.0, 1.0] array in sRGB.
        checkpoint_path: Optional path to .pth checkpoint weights.

    Returns:
        Enhanced HxWx3 float32 array in [0.0, 1.0].
    """
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() > 1.05:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    h, w, c = arr.shape
    # Pad to multiple of 16
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(_MODEL_DEVICE)

    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    model = get_model(checkpoint_path)

    with torch.inference_mode():
        out_tensor = model(tensor)

    # Crop padding away
    out_tensor = out_tensor[:, :, :h, :w]
    out_arr = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Bounded sRGB output without any GT-mean or reference-dependent correction
    return np.clip(out_arr, 0.0, 1.0).astype(np.float32)
