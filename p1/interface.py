"""
Shared Interface for P1 Low-Light Restoration (Freeze at H+1).

Implements:
    enhance(rgb_uint8: HxWx3, config) -> EnhancementResult
with fields:
    - image: HxWx3 sRGB uint8
    - maps: dict[str, np.ndarray] of HxW float32 arrays derived from input
    - params: dict[str, Any] of actual selected settings
    - elapsed_ms: float
    - branch_version: str
    - config_hash: str
    - checkpoint_hash: Optional[str] (None for Part A)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps


def compute_bytes_sha256(data: bytes) -> str:
    """Return hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Return hex SHA-256 of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Deterministic SHA-256 of a JSON-serializable configuration dict."""
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_oriented_rgb_uint8(path: Union[str, Path]) -> Tuple[np.ndarray, str]:
    """
    Decode an image and apply EXIF orientation once before enhance().
    Handles grayscale and RGBA (explicit compositing over black) and returns
    (rgb_uint8: HxWx3 uint8, raw_file_sha256: str).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")
    raw_bytes = p.read_bytes()
    file_sha256 = compute_bytes_sha256(raw_bytes)

    with Image.open(p) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (0, 0, 0))
            bg.paste(img, mask=img.split()[3])
            rgb_img = bg
        elif img.mode != "RGB":
            rgb_img = img.convert("RGB")
        else:
            rgb_img = img
        arr = np.asarray(rgb_img, dtype=np.uint8)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Decoded image must be HxWx3 uint8, got {arr.shape}")
    return np.ascontiguousarray(arr), file_sha256


@dataclass
class EnhancementConfig:
    """Versioned configuration for Part A (classical), Part B (deep), and Part C (hybrid)."""

    mode: str = "deep"  # "identity", "gamma", "classical_fixed", "classical_adaptive", "deep", "hybrid_5050", "hybrid_illum", "hybrid_guided", "hybrid_guarded"
    checkpoint_path: Optional[str] = "weights/cidnet/CIDNETweights/lolv2/best_PSNR.pth"
    device: str = "auto"  # "auto", "cuda", "cpu"
    use_amp: bool = False
    tile_size: int = 0  # 0 = full-frame with factor-8 reflect padding; >0 = deterministic tiled inference
    tile_overlap: int = 32
    alpha_s: float = 1.0  # HVI saturation gating factor (1.0 = unmodified)
    alpha_i: float = 1.0  # HVI intensity scaling factor (1.0 = unmodified, no GT mean)
    gamma_input: float = 1.0
    seed: int = 42
    # Classical & fusion settings
    classical_adaptive: bool = True
    quality_guard: bool = True
    fusion_rule_id: str = "R1_default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self.to_dict())


@dataclass
class EnhancementResult:
    """Standardized return object from enhance(rgb_uint8, config)."""

    image: np.ndarray  # HxWx3 sRGB uint8
    maps: Dict[str, np.ndarray] = field(default_factory=dict)  # named HxW float32 arrays
    params: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    branch_version: str = "p1-v1.0"
    config_hash: str = ""
    checkpoint_hash: Optional[str] = None


def enhance(
    rgb_uint8: np.ndarray,
    config: Optional[Union[Dict[str, Any], EnhancementConfig]] = None,
) -> EnhancementResult:
    """
    Unified entry point for P1 enhancement across Part A, Part B, and Part C.
    Strictly enforces:
      - Input validation (HxWx3 uint8, finite)
      - Reference isolation (no reference image or GT statistics accepted)
      - Exact output dimensions and channel order matching input
    """
    if rgb_uint8 is None or not isinstance(rgb_uint8, np.ndarray):
        raise ValueError("rgb_uint8 must be a non-null numpy ndarray.")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {rgb_uint8.shape}.")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype at interface boundary, got {rgb_uint8.dtype}.")
    if rgb_uint8.shape[0] < 8 or rgb_uint8.shape[1] < 8:
        raise ValueError(f"Image dimensions too small ({rgb_uint8.shape[:2]}); minimum 8x8 required.")

    if config is None:
        cfg = EnhancementConfig()
    elif isinstance(config, dict):
        valid_keys = {f.name for f in EnhancementConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        cfg = EnhancementConfig(**filtered)
    elif isinstance(config, EnhancementConfig):
        cfg = config
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    from .pipeline import run_enhancement_pipeline

    return run_enhancement_pipeline(rgb_uint8, cfg)
