"""Calibration-free affine lifting, fixed Fourier features and the geometry of the SigLIP token grid.

Proposition 1: for an ideal rectified pinhole rig, lifting a pixel ``(u, v)`` with its exact disparity ``d`` under
*any* assumed intrinsics ``(f', c')`` and scale ``s'`` gives ``X' = A X`` with ``A`` a fixed invertible linear map of
the true coordinates ``X``.  The assumed parameters are constants: they define a consistent coordinate frame, not
the metric scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class AffineLifter(nn.Module):
    """Maps native-pixel ``(u, v, d)`` to lifted coordinates ``X'`` (Eq. 3 of the paper).

    ``Z' = s' / (d / W)``, ``X' = (u - c'_x) / f' * Z'``, ``Y' = (v - c'_y) / f' * Z'``.  The disparity is normalised
    by the image width so that ``s'`` does not depend on the input resolution; the principal point is the image
    centre and ``f' = W / (2 tan(fov / 2))``.  All quantities are fixed buffers.
    """

    def __init__(self, fov_deg: float = 60.0, scale: float = 0.03, min_disp_norm: float = 1e-4):
        super().__init__()
        f_norm = 1.0 / (2.0 * math.tan(math.radians(fov_deg) / 2.0))  # f' / W
        # constants (recomputed from the config at construction, hence not part of the checkpoint)
        self.register_buffer("f_norm", torch.tensor(float(f_norm)), persistent=False)
        self.register_buffer("cx_norm", torch.tensor(0.5), persistent=False)  # c'_x / W
        self.register_buffer("cy_norm", torch.tensor(0.5), persistent=False)  # c'_y / H
        self.register_buffer("scale", torch.tensor(float(scale)), persistent=False)
        self.min_disp_norm = float(min_disp_norm)  # delta_d, the numerical floor for small disparities

    def forward(self, u: Tensor, v: Tensor, d: Tensor, width: float, height: float) -> Tensor:
        """u, v, d: broadcastable tensors in native pixels.  Returns ``[..., 3]``."""
        f = self.f_norm * width
        cx = self.cx_norm * width
        cy = self.cy_norm * height
        d_norm = (d / width).clamp_min(self.min_disp_norm)
        z = self.scale / d_norm
        x = (u - cx) / f * z
        y = (v - cy) / f * z
        return torch.stack((x, y, z), dim=-1)


class FourierFeatures(nn.Module):
    """Fixed log-spaced Fourier features of 3D coordinates: R^3 -> R^{6 * bands}."""

    def __init__(self, bands: int = 8, max_log2_freq: float = 4.0):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0.0, max_log2_freq, bands) * (math.pi / 2.0)
        self.register_buffer("freqs", freqs, persistent=False)  # fixed, recomputed from the config
        self.out_dim = 6 * bands

    def forward(self, x: Tensor) -> Tensor:
        proj = (x[..., :, None] * self.freqs).flatten(-2)  # [..., 3 * bands]
        return torch.cat((proj.sin(), proj.cos()), dim=-1)


def quarter_grid_native_coords(h: int, w: int, ffs_scale: float, device, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """Native-pixel centre coordinates ``(u, v)`` of the FFS quarter-resolution grid cells -> two ``[h, w]`` tensors."""
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * 4.0 / ffs_scale
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * 4.0 / ffs_scale
    v, u = torch.meshgrid(ys, xs, indexing="ij")
    return u, v


def resize_with_pad_geometry(native_h: int, native_w: int, canvas: int = 224) -> tuple[int, int, int, int]:
    """Replicates pi0.5's ``resize_with_pad``: returns ``(resized_h, resized_w, pad_top, pad_left)``."""
    ratio = max(native_w / canvas, native_h / canvas)
    rh, rw = int(native_h / ratio), int(native_w / ratio)
    pad_h0 = (canvas - rh) // 2
    pad_w0 = (canvas - rw) // 2
    return rh, rw, pad_h0, pad_w0


def pool_to_token_grid(
    feat: Tensor, conf: Tensor, native_hw: tuple[int, int], canvas: int = 224, patch: int = 14
) -> tuple[Tensor, Tensor]:
    """Confidence-weighted pooling of an FFS-grid map onto the SigLIP token grid (Eq. 5).

    ``feat`` [B,C,h,w], ``conf`` [B,1,h,w] in [0, 1].  The map is resampled onto the 224 canvas with the exact
    resize-with-pad geometry of pi0.5 and averaged over 14x14 patches.  Returns the pooled features [B,C,G,G] and
    the pooled confidence [B,1,G,G] (zero in padded regions and wherever no cell has confidence).
    """
    b, c, _, _ = feat.shape
    rh, rw, pad_h0, pad_w0 = resize_with_pad_geometry(*native_hw, canvas)
    weighted = F.interpolate(feat * conf, size=(rh, rw), mode="bilinear", align_corners=False)
    weight = F.interpolate(conf, size=(rh, rw), mode="bilinear", align_corners=False)
    pad = (pad_w0, canvas - rw - pad_w0, pad_h0, canvas - rh - pad_h0)
    weighted = F.pad(weighted, pad)
    weight = F.pad(weight, pad)
    pooled_w = F.avg_pool2d(weighted, patch)
    pooled_c = F.avg_pool2d(weight, patch)
    pooled = pooled_w / pooled_c.clamp_min(1e-6)
    return pooled, pooled_c


def cell_token_index(
    h: int, w: int, ffs_scale: float, native_hw: tuple[int, int], canvas: int = 224, patch: int = 14, device=None
) -> Tensor:
    """Index ``i(j)`` of the SigLIP token (row-major, G*G) whose patch contains each FFS cell -> ``[h*w]`` long.

    Uses the same resize-with-pad geometry as :func:`pool_to_token_grid`, so a cell and the token it is assigned to
    cover the same native pixels.
    """
    u, v = quarter_grid_native_coords(h, w, ffs_scale, device)
    _, _, pad_h0, pad_w0 = resize_with_pad_geometry(*native_hw, canvas)
    ratio = max(native_hw[1] / canvas, native_hw[0] / canvas)
    g = canvas // patch
    col = ((u / ratio + pad_w0) // patch).clamp(0, g - 1)
    row = ((v / ratio + pad_h0) // patch).clamp(0, g - 1)
    return (row * g + col).long().flatten()


def warp_grid_to_raw(maps: Tensor, H_left: Tensor, ffs_scale: float) -> Tensor:
    """Resample FFS-grid maps [B,C,h,w] defined on the *rectified* left image onto the same grid of the *stored*
    left image (the one SigLIP encodes).  ``H_left`` (3x3, native pixels) maps stored -> rectified coordinates.
    Cells whose source falls outside the rectified image get 0, so a warped confidence map excludes them from the
    confidence-weighted pooling.
    """
    b, _, h, w = maps.shape
    u, v = quarter_grid_native_coords(h, w, ffs_scale, maps.device, dtype=torch.float32)
    p = torch.stack((u, v, torch.ones_like(u)), -1).reshape(-1, 3)  # stored-frame cell centres
    q = p @ H_left.to(device=maps.device, dtype=torch.float32).T
    q = q[:, :2] / q[:, 2:3].clamp_min(1e-8)  # rectified-frame source positions (native px)
    c = q * ffs_scale / 4.0  # continuous cell coordinates in the rectified grid
    gx = 2.0 * c[:, 0] / w - 1.0
    gy = 2.0 * c[:, 1] / h - 1.0
    grid = torch.stack((gx, gy), -1).reshape(1, h, w, 2).expand(b, h, w, 2).to(maps.dtype)
    return F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
