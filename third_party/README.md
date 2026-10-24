# Third-party code

`scripts/setup_third_party.sh` populates this directory:

* `lerobot/` — [LeRobot](https://github.com/huggingface/lerobot) at commit `e40b58a8` (the v0.6.1 line with the
  RoboCasa365 benchmark integration) with `lerobot_stereolift.patch` applied.  The patch adds the task-balanced
  sampler of the RoboCasa365 recipe, a normalisation-statistics override, a camera whitelist
  (`--selected_image_features`) and stricter observation checks in the RoboCasa environment.  It does not touch any
  policy code: StereoLift registers itself as `policy.type=pi05_stereolift` at import time.
* `Fast-FoundationStereo/` — [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) (the frozen stereo
  model) with the released `c-ffs` weights from
  [`nvidia/c-fast-foundationstereo`](https://huggingface.co/nvidia/c-fast-foundationstereo) in `weights/c-ffs/`
  (`model_best_bp2_serialize.pth`, md5 `37459c57fd00c61bedf409f27ee10756`, 71 MB).

Both are ignored by git.
