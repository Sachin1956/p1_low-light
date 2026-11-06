"""
Unit Tests for Deep Learning Branch (HVI-CIDNet).
Tests color space conversion invertibility, network dimensions, and inference wrapper.
"""

import numpy as np
import pytest
import torch

from models.hvi_cidnet import CIDNet, hvi_to_rgb, rgb_to_hvi
from models.inference import enhance_deep


def test_hvi_invertibility():
    """Verify RGB -> HVI -> RGB roundtrip has numerical MSE < 1e-6."""
    rng = np.random.default_rng(seed=1956)
    rgb_arr = rng.uniform(0.0, 1.0, size=(2, 3, 64, 64)).astype(np.float32)
    rgb_tensor = torch.from_numpy(rgb_arr)

    hvi = rgb_to_hvi(rgb_tensor)
    rgb_rec = hvi_to_rgb(hvi)

    mse = torch.mean((rgb_tensor - rgb_rec) ** 2).item()
    assert mse < 1e-6, f"HVI roundtrip MSE too large: {mse}"


def test_cidnet_forward_pass():
    """Verify CIDNet produces same shape, finite output in [0, 1]."""
    model = CIDNet(base_channels=16)
    model.eval()

    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_enhance_deep_arbitrary_dimensions():
    """Verify enhance_deep pads and crops cleanly for non-multiple-of-16 resolutions."""
    # Arbitrary non-power-of-2, prime-like dimensions
    odd_img = np.random.randint(10, 80, size=(137, 219, 3), dtype=np.uint8)
    out = enhance_deep(odd_img)

    assert isinstance(out, np.ndarray)
    assert out.shape == (137, 219, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
