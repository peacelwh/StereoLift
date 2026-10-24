"""StereoLift: calibration-free stereo lifting for vision-language-action models (pi0.5 + Fast-FoundationStereo)."""

from . import lerobot_plugin as _plugin  # noqa: F401  (registers policy.type=pi05_stereolift)

__version__ = "1.0.0"
__all__ = ["__version__"]
