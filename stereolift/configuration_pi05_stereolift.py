#!/usr/bin/env python
"""Configuration of ``policy.type=pi05_stereolift``: pi0.5 conditioned on the disparity belief of a frozen
Fast-FoundationStereo (FFS) model through a geometric token embedding and geometric relation tokens.

Every field below is a ``--policy.<name>=<value>`` option of ``python -m stereolift.train`` and ``stereolift.eval``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

RECTIFY_MODES = ("none", "vertical_shift", "homography")

_REPO_ROOT = str(Path(__file__).resolve().parents[1])  # the StereoLift checkout this module is imported from
DEFAULT_FFS_DIR = os.environ.get("STEREOLIFT_FFS_DIR", f"{_REPO_ROOT}/third_party/Fast-FoundationStereo")
DEFAULT_FFS_CHECKPOINT = os.environ.get(
    "STEREOLIFT_FFS_CHECKPOINT", f"{_REPO_ROOT}/third_party/Fast-FoundationStereo/weights/c-ffs/model_best_bp2_serialize.pth"
)
_PORTABLE_PATH_FIELDS = ("ffs_repo_dir", "ffs_checkpoint", "rectify_homography_path")


def _relocate(path: str | None) -> str | None:
    """Map a path inside another machine's StereoLift checkout (stored in a checkpoint config) to this checkout."""
    if not path or os.path.exists(path):
        return path
    for marker in ("/third_party/", "/configs/"):
        if marker in path:
            alt = os.path.join(_REPO_ROOT, marker.strip("/"), path.split(marker, 1)[1])
            if os.path.exists(alt):
                return alt
    return path


@PreTrainedConfig.register_subclass("pi05_stereolift")
@dataclass
class PI05StereoLiftConfig(PI05Config):
    """pi0.5 + StereoLift."""

    # ------------------------------------------------------------------ inputs
    # The rectified-or-rectifiable stereo pair.  The left image is encoded by SigLIP; both images go to FFS.
    stereo_left_key: str = "observation.images.head_left"
    stereo_right_key: str = "observation.images.head_right"
    state_key: str = "observation.state"

    # -------------------------------------------------------- stereo backbone
    ffs_repo_dir: str = DEFAULT_FFS_DIR
    ffs_checkpoint: str = DEFAULT_FFS_CHECKPOINT
    ffs_input_scale: float = 1.0  # 1.0 for 256x256 simulation renders; 0.5 for the 640x480 robot cameras
    ffs_iters: int = 4  # ConvGRU refinement iterations K (a test-time knob)
    ffs_max_disp: int = 192  # disparity range d_max in pixels at the FFS input resolution
    # Ablation D (Table 3): collapse the belief to the refined disparity d_K (no moments, state, or confidence).
    belief_point_only: bool = False

    # ---------------------------------------------------------- rectification
    rectify_mode: str = "none"  # none | vertical_shift (robot) | homography (RoboCasa toe-in pair)
    rectify_right_dx: float = 0.0  # vertical_shift: translation of the right image in native pixels
    rectify_right_dy: float = 0.0
    rectify_homography_path: str | None = None  # homography: json with H_left, H_right (3x3, native pixels)
    # homography mode only: FFS runs on the rectified pair while SigLIP encodes the stored left image, so the belief
    # maps are resampled onto the stored left frame before they are pooled onto the token grid.
    align_geometry_to_raw: bool = True

    # ------------------------------------------------------------- lifting
    lift_fov_deg: float = 60.0  # assumed horizontal field of view -> f' = W / (2 tan(fov / 2)); c' = image centre
    lift_scale: float = 0.03  # assumed scale s' (Z' = s' / (d / W))
    lift_fourier_bands: int = 8  # fixed log-spaced Fourier bands per axis
    lift_fourier_max_log2: float = 4.0  # highest band = 2^max_log2 * pi / 2 radians per lifted unit

    # ------------------------------------------------------------- interface
    adapter_hidden_dim: int = 256
    adapter_num_heads: int = 4
    adapter_dropout: float = 0.0
    use_geometric_embedding: bool = True  # ablation C: False removes the geometric token embedding
    num_relation_tokens: int = 16  # M; ablation B: 0 removes the relation tokens
    relation_semantic_keys: bool = True  # keys carry W_t t_{i(j)}, the token of the patch containing the cell
    relation_use_state: bool = True  # FiLM on the normalised proprioceptive state (in addition to language)
    relation_max_keys: int = 8192  # subsample cells beyond this many keys (memory guard for large inputs)
    key_penalty: float = 4.0  # lambda in  logit_mj -= lambda (1 - c_j)
    adapter_lr_scale: float = 10.0  # learning-rate multiplier of the added interface parameters
    freeze_policy_backbone: bool = False  # train the interface only (smoke tests)

    def __post_init__(self):
        super().__post_init__()
        for name in _PORTABLE_PATH_FIELDS:
            setattr(self, name, _relocate(getattr(self, name)))
        if self.stereo_left_key == self.stereo_right_key:
            raise ValueError("stereo_left_key and stereo_right_key must differ")
        if self.rectify_mode not in RECTIFY_MODES:
            raise ValueError(f"rectify_mode must be one of {RECTIFY_MODES}, got {self.rectify_mode!r}")
        if self.rectify_mode == "homography" and not self.rectify_homography_path:
            raise ValueError("rectify_mode=homography requires rectify_homography_path")
        if not 0 < self.ffs_input_scale <= 1.0:
            raise ValueError("ffs_input_scale must be in (0, 1]")
        if self.ffs_iters < 0:
            raise ValueError("ffs_iters must be non-negative")
        if self.ffs_max_disp % 4:
            raise ValueError("ffs_max_disp must be divisible by 4")
        if self.adapter_hidden_dim % self.adapter_num_heads:
            raise ValueError("adapter_hidden_dim must be divisible by adapter_num_heads")
        if self.num_relation_tokens < 0:
            raise ValueError("num_relation_tokens must be non-negative")
        if self.lift_scale <= 0 or not 0 < self.lift_fov_deg < 180:
            raise ValueError("lift_scale must be positive and lift_fov_deg in (0, 180)")
        if self.adapter_lr_scale <= 0:
            raise ValueError("adapter_lr_scale must be positive")

    def validate_features(self) -> None:
        super().validate_features()
        keys = list(self.image_features)
        for key in (self.stereo_left_key, self.stereo_right_key):
            if key not in keys:
                raise ValueError(
                    f"pi05_stereolift requires image feature {key!r}; available image features: {keys}. "
                    "Set --selected_image_features accordingly."
                )

    @property
    def siglip_image_keys(self) -> list[str]:
        """Image keys encoded by SigLIP, in prefix order: the left view first, then any auxiliary cameras.

        The right view contributes through the stereo module only, never as a second set of visual tokens.
        """
        keys = [self.stereo_left_key]
        keys += [k for k in self.image_features if k not in (self.stereo_left_key, self.stereo_right_key)]
        return keys
