"""``python -m stereolift.train <lerobot-train args>``: lerobot-train with the StereoLift policy registered."""

from __future__ import annotations

import stereolift  # noqa: F401  (registers the policy type)
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
