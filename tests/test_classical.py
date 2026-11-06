"""
Comprehensive Test Suite for Part A: Classical Low-Light Enhancement.
Tests all requirements and acceptance gates from:
- docs/P1_AGENT_CONTRACT.md
- docs/P1_PART_A_IMPLEMENTATION_PLAN.md
"""

import hashlib
import time
import numpy as np
import pytest

from classical.config import ClassicalConfig, EnhancementResult, PipelineMode
from classical.features import (
    compute_diagnostic_luminance,
    compute_edge_gradient_map,
    compute_highlight_protection_mask,
    estimate_flat_shadow_noise,
    extract_features_and_maps,
)
from classical.mappings import clip01, map_features_to_parameters
from classical.pipeline import enhance, validate_input_image


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sample_dark_image() -> np.ndarray:
    """Create a controlled 256x256 dark synthetic image with textured shadows and a bright lamp."""
    rng = np.random.default_rng(seed=1956)
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # 1. Dark background with slight sensor noise
    noise = rng.normal(loc=20.0, scale=4.0, size=(256, 256, 3))
    img = np.clip(noise, 5, 45).astype(np.uint8)

    # 2. Midtone textured patch (e.g. wall/furniture)
    img[60:180, 40:120, :] = rng.integers(30, 70, size=(120, 80, 3), dtype=np.uint8)

    # 3. Bright lamp / specular highlight region (to test highlight guard)
    img[30:70, 190:230, :] = rng.integers(235, 255, size=(40, 40, 3), dtype=np.uint8)

    return img


@pytest.fixture
def sample_bright_image() -> np.ndarray:
    """Create a well-exposed / bright indoor image."""
    rng = np.random.default_rng(seed=2026)
    img = rng.integers(140, 220, size=(200, 200, 3), dtype=np.uint8)
    return img


# =========================================================================
# Gate 0: Interface and Determinism Checks
# =========================================================================

def test_interface_compliance(sample_dark_image):
    """Verify shared interface signature, output types, dimensions, and map keys."""
    res = enhance(sample_dark_image)

    assert isinstance(res, EnhancementResult)
    assert isinstance(res.image, np.ndarray)
    assert res.image.dtype == np.uint8
    assert res.image.shape == sample_dark_image.shape
    assert res.checkpoint_hash is None
    assert isinstance(res.config_hash, str) and len(res.config_hash) == 16
    assert isinstance(res.elapsed_ms, float) and res.elapsed_ms > 0

    # Verify required diagnostic maps
    required_maps = [
        "illumination_raw",
        "illumination_refined",
        "luminance_diagnostic",
        "edge_map",
        "highlight_mask",
        "flat_shadow_validity",
    ]
    for k in required_maps:
        assert k in res.maps, f"Missing required map: {k}"
        m = res.maps[k]
        assert isinstance(m, np.ndarray)
        assert m.dtype == np.float32
        assert m.shape == sample_dark_image.shape[:2]
        assert not np.isnan(m).any(), f"NaN in map {k}"
        assert not np.isinf(m).any(), f"Inf in map {k}"
        assert m.min() >= -1e-5 and m.max() <= 1.00001, f"Map {k} values out of bounds [0, 1]"


def test_deterministic_reproducibility(sample_dark_image):
    """Verify two runs of identical input produce byte-for-byte identical output."""
    cfg = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)
    res1 = enhance(sample_dark_image, cfg)
    res2 = enhance(sample_dark_image, cfg)

    h1 = hashlib.sha256(res1.image.tobytes()).hexdigest()
    h2 = hashlib.sha256(res2.image.tobytes()).hexdigest()
    assert h1 == h2, "Nondeterministic output detected for identical input!"


# =========================================================================
# Gate 1: E0 vs E1 vs E2 Verification
# =========================================================================

def test_e0_baseline(sample_dark_image):
    """Test E0 baseline mode (fixed gamma adjustment)."""
    cfg_e0 = ClassicalConfig(mode=PipelineMode.E0_BASELINE, e0_gamma=0.75)
    res_e0 = enhance(sample_dark_image, cfg_e0)

    assert res_e0.params["applied_mode"] == "E0_BASELINE"
    assert res_e0.params["gamma"] == 0.75
    # Output must be brighter than input in dark regions
    assert np.mean(res_e0.image) > np.mean(sample_dark_image)


def test_e1_fixed_classical(sample_dark_image):
    """Test E1 fixed classical mode satisfies all required operations."""
    cfg_e1 = ClassicalConfig(mode=PipelineMode.E1_FIXED)
    res_e1 = enhance(sample_dark_image, cfg_e1)

    assert res_e1.params["applied_mode"] == "E1_FIXED"
    assert res_e1.params["gamma"] == cfg_e1.fixed_gamma
    assert res_e1.params["gain_cap"] == cfg_e1.fixed_gain_cap
    assert res_e1.params["clahe_clip_limit"] == cfg_e1.fixed_clahe_clip
    assert res_e1.params["denoise_strength"] == cfg_e1.fixed_denoise_strength
    assert np.mean(res_e1.image) > np.mean(sample_dark_image)


# =========================================================================
# Gate 2: Concrete Monotone Mappings
# =========================================================================

def test_monotone_mappings_sweep():
    """
    Verify mathematical monotonicity of E2 mappings:
    - Decreasing median luminance m -> strictly decreasing gamma (more lift)
    - Increasing highlight fraction h -> strictly decreasing gain cap
    - Increasing noise n -> strictly increasing denoise and decreasing CLAHE clip
    """
    from classical.features import ImageAnalysisFeatures

    cfg = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)

    # 1. Luminance sweep: m from 0.05 to 0.60
    m_values = [0.05, 0.10, 0.20, 0.35, 0.50, 0.60]
    gammas = []
    for m in m_values:
        feat = ImageAnalysisFeatures(
            median_luminance=m,
            highlight_fraction=0.01,
            noise_estimate=0.02,
            shadow_patch_fraction=0.10,
            noise_is_valid=True,
            mean_luminance=m,
            dynamic_range=0.5,
        )
        p = map_features_to_parameters(feat, cfg)
        gammas.append(p.gamma)

    # Gamma should be monotonically non-decreasing as m increases (less lift)
    for i in range(len(gammas) - 1):
        assert gammas[i] <= gammas[i + 1] + 1e-6, f"Gamma not monotonic: {gammas}"

    # 2. Highlight sweep: h from 0.0 to 0.15
    h_values = [0.0, 0.02, 0.05, 0.08, 0.12]
    gain_caps = []
    for h in h_values:
        feat = ImageAnalysisFeatures(
            median_luminance=0.10,
            highlight_fraction=h,
            noise_estimate=0.02,
            shadow_patch_fraction=0.10,
            noise_is_valid=True,
            mean_luminance=0.10,
            dynamic_range=0.9,
        )
        p = map_features_to_parameters(feat, cfg)
        gain_caps.append(p.gain_cap)

    # Gain cap should be monotonically non-increasing as highlight fraction increases
    for i in range(len(gain_caps) - 1):
        assert gain_caps[i] >= gain_caps[i + 1] - 1e-6, f"Gain cap not monotonic: {gain_caps}"

    # 3. Noise sweep: n from 0.005 to 0.08
    n_values = [0.005, 0.015, 0.030, 0.050, 0.070]
    denoises, clips = [], []
    for n in n_values:
        feat = ImageAnalysisFeatures(
            median_luminance=0.15,
            highlight_fraction=0.01,
            noise_estimate=n,
            shadow_patch_fraction=0.10,
            noise_is_valid=True,
            mean_luminance=0.15,
            dynamic_range=0.5,
        )
        p = map_features_to_parameters(feat, cfg)
        denoises.append(p.denoise_strength)
        clips.append(p.clahe_clip_limit)

    for i in range(len(denoises) - 1):
        assert denoises[i] <= denoises[i + 1] + 1e-6, f"Denoise not monotonic: {denoises}"
        assert clips[i] >= clips[i + 1] - 1e-6, f"CLAHE clip not monotonic: {clips}"


def test_invalid_noise_fallback():
    """Verify that when q < 0.02, noise adaptation cleanly falls back to fixed defaults."""
    from classical.features import ImageAnalysisFeatures

    cfg = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)
    feat_invalid = ImageAnalysisFeatures(
        median_luminance=0.15,
        highlight_fraction=0.01,
        noise_estimate=0.08,  # high noise, but invalid mask!
        shadow_patch_fraction=0.01, # q < 0.02
        noise_is_valid=False,
        mean_luminance=0.15,
        dynamic_range=0.5,
    )
    p = map_features_to_parameters(feat_invalid, cfg)

    assert p.noise_adapted is False
    assert p.denoise_strength == cfg.fixed_denoise_strength
    assert p.clahe_clip_limit == cfg.fixed_clahe_clip


def test_controlled_inputs_select_different_parameters(sample_dark_image, sample_bright_image):
    """Verify that at least two controlled inputs select different parameters."""
    cfg = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)
    res_dark = enhance(sample_dark_image, cfg)
    res_bright = enhance(sample_bright_image, cfg)

    # Dark image must select lower gamma (stronger lift) and higher gain cap
    assert res_dark.params["gamma"] < res_bright.params["gamma"]
    assert res_dark.params["gain_cap"] > res_bright.params["gain_cap"]


# =========================================================================
# Gate 3: Ablation Sensitivity Checks
# =========================================================================

def test_ablation_illumination_refinement(sample_dark_image):
    """Ablation: varying refinement radius changes output."""
    cfg1 = ClassicalConfig(illumination_refinement_radius=5)
    cfg2 = ClassicalConfig(illumination_refinement_radius=25)
    res1 = enhance(sample_dark_image, cfg1)
    res2 = enhance(sample_dark_image, cfg2)
    assert not np.array_equal(res1.image, res2.image)


def test_ablation_denoise(sample_dark_image):
    """Ablation: disabling denoiser changes output on noisy dark fixture."""
    cfg_on = ClassicalConfig(enable_denoise=True)
    cfg_off = ClassicalConfig(enable_denoise=False)
    res_on = enhance(sample_dark_image, cfg_on)
    res_off = enhance(sample_dark_image, cfg_off)
    assert not np.array_equal(res_on.image, res_off.image)


def test_ablation_clahe(sample_dark_image):
    """Ablation: disabling CLAHE changes output."""
    cfg_on = ClassicalConfig(enable_clahe=True)
    cfg_off = ClassicalConfig(enable_clahe=False)
    res_on = enhance(sample_dark_image, cfg_on)
    res_off = enhance(sample_dark_image, cfg_off)
    assert not np.array_equal(res_on.image, res_off.image)


def test_ablation_highlight_guard(sample_dark_image):
    """Ablation: highlight guard preserves lamp highlight without blowout."""
    cfg_guard_on = ClassicalConfig(enable_highlight_guard=True)
    cfg_guard_off = ClassicalConfig(enable_highlight_guard=False)

    res_on = enhance(sample_dark_image, cfg_guard_on)
    res_off = enhance(sample_dark_image, cfg_guard_off)

    # Lamp region in sample_dark_image is [30:70, 190:230]
    lamp_orig = sample_dark_image[30:70, 190:230]
    lamp_on = res_on.image[30:70, 190:230]
    lamp_off = res_off.image[30:70, 190:230]

    # Without guard, Retinex + CLAHE pushes highlights higher or changes contrast
    assert not np.array_equal(lamp_on, lamp_off)
    # With guard, the lamp stays closer to original input
    diff_on = np.mean(np.abs(lamp_on.astype(float) - lamp_orig.astype(float)))
    diff_off = np.mean(np.abs(lamp_off.astype(float) - lamp_orig.astype(float)))
    assert diff_on <= diff_off + 1e-4


# =========================================================================
# Robustness & Adversarial Checks
# =========================================================================

def test_extreme_black_image():
    """Verify that all-zero image does not crash or produce NaNs."""
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    res = enhance(black)
    assert res.image.shape == black.shape
    assert not np.isnan(res.image).any()


def test_extreme_white_image():
    """Verify that all-255 image does not crash or produce NaNs."""
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = enhance(white)
    assert res.image.shape == white.shape
    assert not np.isnan(res.image).any()


def test_invalid_input_shapes():
    """Verify explicit ValueError on bad dimensions."""
    with pytest.raises(ValueError):
        enhance(np.zeros((100, 100), dtype=np.uint8))  # 2D
    with pytest.raises(ValueError):
        enhance(np.zeros((100, 100, 4), dtype=np.uint8))  # 4-channel
    with pytest.raises(ValueError):
        enhance(np.zeros((1, 1, 3), dtype=np.uint8))  # too small


def test_cpu_latency_benchmark(sample_dark_image):
    """Measure inference latency: must finish well under 1 second on CPU for 256x256."""
    cfg = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)
    # Warmup
    _ = enhance(sample_dark_image, cfg)

    # 5 runs
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = enhance(sample_dark_image, cfg)
        times.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(times))
    # For 256x256, typical CPU runtime is < 60ms, well below the 1000ms target
    assert median_ms < 500.0, f"Classical inference too slow: {median_ms:.2f} ms"
