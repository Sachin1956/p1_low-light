"""
Unit Tests for Part C: Adaptive Spatial Fusion.
Tests all requirements and acceptance gates from docs/P1_ML_EXECUTION_PLAN.md Section 4.
"""

import numpy as np
import pytest

from fusion.config import FusionConfig, FusionResult
from fusion.spatial_fusion import (
    apply_quality_awareness_guard,
    compute_smooth_candidate_weights,
    fuse_adaptive,
    fuse_hybrid,
)


@pytest.fixture
def mixed_light_scene():
    """Create a 200x200 scene with dark shadows, midtones, and a bright specular highlight."""
    low = np.zeros((200, 200, 3), dtype=np.uint8)
    low[:100, :100, :] = 15      # deep darkness
    low[:100, 100:, :] = 80      # midtone
    low[100:, :100, :] = 40      # dark texture
    low[100:, 100:, :] = 240     # bright specular highlight

    # Mock classical enhancement
    classical = np.copy(low).astype(float) * 2.0
    classical = np.clip(classical, 0, 255).astype(np.uint8)

    # Mock deep learning enhancement
    deep = np.copy(low).astype(float) * 2.5
    deep = np.clip(deep, 0, 255).astype(np.uint8)

    return low, classical, deep


def test_spatial_weights_normalized(mixed_light_scene):
    """Verify weights are nonnegative and sum to 1.0 within 1e-5 everywhere."""
    low, classical, deep = mixed_light_scene
    cfg = FusionConfig()
    res = fuse_adaptive(low, classical, deep, cfg)

    assert isinstance(res, FusionResult)
    weights = res.weights  # HxWx3
    assert np.all(weights >= 0.0), "Negative weights detected!"
    weight_sum = np.sum(weights, axis=-1)
    diff = np.abs(weight_sum - 1.0)
    assert np.all(diff < 1e-5), f"Weights do not sum to 1.0 (max diff: {np.max(diff)})"


def test_spatial_weight_distribution(mixed_light_scene):
    """
    Verify intended spatial behavior:
    - W_o is highest in the bright highlight region (100:, 100:)
    - W_d is highest in the deep dark region (:100, :100)
    """
    low, classical, deep = mixed_light_scene
    res = fuse_adaptive(low, classical, deep)
    weights = res.weights

    # Dark patch: (:50, :50)
    w_dark = np.mean(weights[:50, :50, :], axis=(0, 1))
    assert w_dark[2] > w_dark[0], f"Deep weight should dominate in darkness: {w_dark}"

    # Bright highlight patch: (150:, 150:)
    w_bright = np.mean(weights[150:, 150:, :], axis=(0, 1))
    assert w_bright[0] > w_bright[2], f"Original weight should dominate in bright region: {w_bright}"


def test_branch_disabling_changes_output(mixed_light_scene):
    """Verify that zeroing out any branch affects the final output."""
    low, classical, deep = mixed_light_scene
    zero_deep = np.zeros_like(deep)
    res_normal = fuse_adaptive(low, classical, deep)
    res_no_deep = fuse_adaptive(low, classical, zero_deep)

    assert not np.array_equal(res_normal.image, res_no_deep.image), "Disabling deep branch had no effect!"


def test_quality_awareness_guard():
    """Verify quality guard prevents artificial blown highlights."""
    low = np.zeros((50, 50, 3), dtype=np.float32) + 0.20  # dark
    # Enhanced output pushes pixels into harsh clipping (> 0.98)
    blown_fused = np.zeros((50, 50, 3), dtype=np.float32) + 0.999

    guarded = apply_quality_awareness_guard(low, blown_fused, clipping_ceiling=0.98)
    assert np.all(guarded <= 0.98), "Quality guard failed to cap artificial blowout!"
    assert np.all(guarded < blown_fused), "Quality guard did not compress clipped region!"


def test_fuse_hybrid_compatibility(mixed_light_scene):
    """Test compatibility with evaluate.py's expected fuse_hybrid signature."""
    low, classical, deep = mixed_light_scene
    out = fuse_hybrid(low, classical, deep)
    assert isinstance(out, np.ndarray)
    assert out.shape == low.shape
