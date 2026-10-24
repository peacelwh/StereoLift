"""``python -m stereolift.eval <lerobot-eval args>``: lerobot-eval with the StereoLift policy registered.

If ``STEREOLIFT_CAM_PERTURB`` is set (a JSON object), the RoboCasa stereo rig is perturbed after every reset
(see :mod:`stereolift.robocasa_perturb`); this is how the camera / baseline robustness studies are run.
"""

from __future__ import annotations

import stereolift  # noqa: F401  (registers the policy type)
from lerobot.scripts.lerobot_eval import main

from stereolift import robocasa_perturb

if __name__ == "__main__":
    robocasa_perturb.install()
    main()
