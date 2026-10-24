"""Light-weight stereo rectification helpers (torch, differentiable, batched).

Two situations are covered:

* Real robot (G1-D head camera): a single 1280x480 frame is split into left/right
  halves.  The pair is approximately rectified but shows a constant vertical offset
  (dataset median -4 px).  ``vertical_shift`` applies a fixed translation to the right
  image.
* Simulation (RoboCasa ``agentview_left/right``): a toe-in pair with a 0.7 m baseline.
  ``homography`` applies fixed per-view homographies obtained from
  ``cv2.stereoRectify`` (see ``scripts/compute_robocasa_rectification.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def translate_image(img: Tensor, dx: float, dy: float) -> Tensor:
    """Translate ``img`` [B,3,H,W] by (dx, dy) pixels (content moves right/down for +)."""
    if dx == 0 and dy == 0:
        return img
    b, _, h, w = img.shape
    theta = img.new_tensor([[1.0, 0.0, -2.0 * dx / max(w - 1, 1)], [0.0, 1.0, -2.0 * dy / max(h - 1, 1)]])
    theta = theta[None].expand(b, 2, 3)
    grid = F.affine_grid(theta, size=(b, 3, h, w), align_corners=True)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def warp_homography(img: Tensor, H: Tensor) -> Tensor:
    """Warp ``img`` [B,3,H,W] with homography ``H`` (3x3, pixel coordinates, output <- H @ input).

    For an output pixel p, the source pixel is H^{-1} p.  Outside pixels are zero.
    """
    b, _, h, w = img.shape
    device, dtype = img.device, img.dtype
    Hinv = torch.linalg.inv(H.to(torch.float64)).to(dtype=dtype, device=device)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij"
    )
    ones = torch.ones_like(xs)
    pts = torch.stack((xs, ys, ones), dim=-1).reshape(-1, 3)  # [HW,3]
    src = pts @ Hinv.T
    src = src[:, :2] / src[:, 2:3].clamp_min(1e-8)
    gx = 2.0 * src[:, 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * src[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1).reshape(1, h, w, 2).expand(b, h, w, 2)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class StereoRectifier:
    """Applies the configured rectification to a (left, right) pair at native resolution."""

    def __init__(self, mode: str, right_dx: float = 0.0, right_dy: float = 0.0, homography_path: str | None = None):
        self.mode = mode
        self.right_dx = float(right_dx)
        self.right_dy = float(right_dy)
        self.H_left: Tensor | None = None
        self.H_right: Tensor | None = None
        self.native_size: tuple[int, int] | None = None
        if mode == "homography":
            data = json.loads(Path(homography_path).read_text())
            self.H_left = torch.tensor(data["H_left"], dtype=torch.float32)
            self.H_right = torch.tensor(data["H_right"], dtype=torch.float32)
            if "size" in data:  # [W, H] the homographies were computed for
                self.native_size = tuple(int(v) for v in data["size"])

    def __call__(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        if self.mode == "none":
            return left, right
        if self.mode == "vertical_shift":
            return left, translate_image(right, self.right_dx, self.right_dy)
        if self.mode == "homography":
            H_l, H_r = self.H_left.to(left.device), self.H_right.to(left.device)
            h, w = left.shape[-2:]
            if self.native_size is not None and (w, h) != self.native_size:
                # rescale pixel homographies to the current resolution
                sx, sy = w / self.native_size[0], h / self.native_size[1]
                S = torch.diag(torch.tensor([sx, sy, 1.0], device=left.device))
                Si = torch.diag(torch.tensor([1.0 / sx, 1.0 / sy, 1.0], device=left.device))
                H_l, H_r = S @ H_l @ Si, S @ H_r @ Si
            return warp_homography(left, H_l), warp_homography(right, H_r)
        raise ValueError(f"Unknown rectification mode {self.mode!r}")


def stereo_rectify_homographies(K1, R1, t1, K2, R2, t2, width: int, height: int):
    """Compute rectifying homographies for two pinhole cameras (numpy, uses OpenCV).

    ``R_i, t_i`` map world -> camera_i (x_cam = R_i x_world + t_i).  Returns
    ``H_left, H_right, P1, P2, baseline`` with H in output <- input pixel coordinates.
    """
    import cv2
    import numpy as np

    R1, R2 = np.asarray(R1, dtype=np.float64), np.asarray(R2, dtype=np.float64)
    t1, t2 = np.asarray(t1, dtype=np.float64).reshape(3), np.asarray(t2, dtype=np.float64).reshape(3)
    # relative pose: x2 = R x1 + T
    R = R2 @ R1.T
    T = t2 - R @ t1
    Rr1, Rr2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        np.asarray(K1, dtype=np.float64), None, np.asarray(K2, dtype=np.float64), None, (width, height), R, T,
        flags=0, alpha=-1, newImageSize=(width, height),
    )
    K1n, K2n = P1[:, :3], P2[:, :3]
    H_left = K1n @ Rr1 @ np.linalg.inv(np.asarray(K1, dtype=np.float64))
    H_right = K2n @ Rr2 @ np.linalg.inv(np.asarray(K2, dtype=np.float64))
    baseline = float(abs(P2[0, 3] / P2[0, 0]))
    return H_left, H_right, P1, P2, baseline
