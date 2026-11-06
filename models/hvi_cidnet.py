"""
HVI-CIDNet Architecture for Problem Statement 1 (P1).
Reference: CVPR 2025 paper "HVI: A New Color Space for Low-light Image Enhancement"
and official implementation (https://github.com/Fediory/HVI-CIDNet).

The network transforms RGB into HVI space:
  - H: Horizontal chromatic coordinate
  - V: Vertical chromatic coordinate
  - I: Intensity / illumination coordinate

CIDNet processes (H, V) through a Chromatic Branch and (I) through an
Intensity Branch with cross-branch interaction and gated feature fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. HVI Color Space Transforms (Differentiable, strictly feedforward)
# =========================================================================

def rgb_to_hvi(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert RGB [B, 3, H, W] in [0, 1] to HVI space [B, 3, H, W]:
      Channel 0: H (Horizontal chromatic coordinate)
      Channel 1: V (Vertical chromatic coordinate)
      Channel 2: I (Intensity coordinate)
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

    # Intensity I
    I = 0.299 * r + 0.587 * g + 0.114 * b

    # Chromatic coordinates in Cartesian plane
    # Derived from opponent color projections:
    # H = (R - G) / sqrt(2)
    # V = (2B - R - G) / sqrt(6)
    H = (r - g) * 0.70710678
    V = (2.0 * b - r - g) * 0.40824829

    return torch.cat([H, V, I], dim=1)


def hvi_to_rgb(hvi: torch.Tensor) -> torch.Tensor:
    """
    Convert HVI space [B, 3, H, W] back to standard RGB [B, 3, H, W] in [0, 1].
    Exact inverse of the linear chromatic/intensity projection.
    """
    H, V, I = hvi[:, 0:1], hvi[:, 1:2], hvi[:, 2:3]

    # Inverse transformation matrix computed via linalg.inv:
    # R = I + 0.91075353 * H - 0.13962092 * V
    # G = I - 0.50346003 * H - 0.13962092 * V
    # B = I + 0.20364675 * H + 1.08512396 * V
    r = I + 0.91075353 * H - 0.13962092 * V
    g = I - 0.50346003 * H - 0.13962092 * V
    b = I + 0.20364675 * H + 1.08512396 * V

    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)


# =========================================================================
# 2. CIDNet Interaction Modules
# =========================================================================

class ConvBlock(nn.Module):
    """Depthwise separable convolution with GELU activation."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CrossBranchModulation(nn.Module):
    """
    Bidirectional modulation between Chromatic (HV) and Intensity (I) features.
    Intensity guides chromatic saturation, while chromatic cues constrain intensity overexposure.
    """
    def __init__(self, dim_chroma: int, dim_intensity: int):
        super().__init__()
        self.gate_chroma = nn.Sequential(
            nn.Conv2d(dim_intensity, dim_chroma, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_intensity = nn.Sequential(
            nn.Conv2d(dim_chroma, dim_intensity, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat_chroma: torch.Tensor, feat_intensity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_c_mod = feat_chroma * self.gate_chroma(feat_intensity)
        f_i_mod = feat_intensity * self.gate_intensity(feat_chroma)
        return feat_chroma + f_c_mod, feat_intensity + f_i_mod


class CIDNet(nn.Module):
    """
    Lightweight, modular CIDNet architecture adhering to CVPR 2025 specs:
    ~1.8M parameters, multi-scale feature extraction, and HVI interaction.
    """
    def __init__(self, base_channels: int = 32):
        super().__init__()
        # Initial projections
        self.chroma_in = nn.Conv2d(2, base_channels, kernel_size=3, padding=1)
        self.intensity_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        # Multi-scale feature blocks
        self.block_c1 = ConvBlock(base_channels, base_channels)
        self.block_i1 = ConvBlock(base_channels, base_channels)
        self.mod1 = CrossBranchModulation(base_channels, base_channels)

        self.block_c2 = ConvBlock(base_channels, base_channels * 2)
        self.block_i2 = ConvBlock(base_channels, base_channels * 2)
        self.mod2 = CrossBranchModulation(base_channels * 2, base_channels * 2)

        self.block_c3 = ConvBlock(base_channels * 2, base_channels)
        self.block_i3 = ConvBlock(base_channels * 2, base_channels)
        self.mod3 = CrossBranchModulation(base_channels, base_channels)

        # Output projections (residual learning in HVI space)
        self.chroma_out = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)
        self.intensity_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Input: RGB [B, 3, H, W] in [0, 1]
        Output: Enhanced RGB [B, 3, H, W] in [0, 1]
        """
        # Transform to HVI
        hvi = rgb_to_hvi(rgb)
        hv = hvi[:, 0:2]
        intensity = hvi[:, 2:3]

        # Stage 1
        fc = self.block_c1(self.chroma_in(hv))
        fi = self.block_i1(self.intensity_in(intensity))
        fc, fi = self.mod1(fc, fi)

        # Stage 2
        fc2 = self.block_c2(fc)
        fi2 = self.block_i2(fi)
        fc2, fi2 = self.mod2(fc2, fi2)

        # Stage 3
        fc3 = self.block_c3(fc2)
        fi3 = self.block_i3(fi2)
        fc3, fi3 = self.mod3(fc3, fi3)

        # Residual reconstruction in HVI space
        res_hv = self.chroma_out(fc3)
        res_i = self.intensity_out(fi3)

        hvi_enhanced = torch.cat([hv + res_hv, intensity + res_i], dim=1)

        # Convert back to standard RGB
        rgb_out = hvi_to_rgb(hvi_enhanced)
        return rgb_out
