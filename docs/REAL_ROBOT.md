# Real robot: humanoid with two dexterous hands

The real-robot experiments use a Unitree G1-D humanoid with two 7-DoF arms and two BrainCo 6-DoF dexterous hands
(26-dimensional action space). Observations are a head-mounted stereo pair (640 × 480 pixels, approximately 0.06 m
baseline and 0.4 m working distance) and two wrist cameras; StereoLift uses the head pair only. The only calibration
correction is a constant vertical offset of the right image estimated from the data (−4.15 pixels on our rig).

## Relevant files

* `scripts/estimate_real_rectification.py`: estimates the residual vertical offset of a stereo pair from a dataset
* `scripts/train_real_robot.sh`: fine-tuning launcher
* `scripts/smoke_test_policy.py`: builds the policy on a few frames and runs training and inference steps
* `stereolift/modeling_pi05_stereolift.py`: `PI05StereoLiftPolicy.last_stereo_confidence` is the mean stereo
  confidence of the last observation, the signal used for failure monitoring

## Dataset format

A LeRobot v3 dataset with

| key | shape | note |
|---|---|---|
| `observation.images.head_left`, `observation.images.head_right` | 480 × 640 × 3 video (640 × 480 pixels), 30 fps | the stereo pair (any other cameras are ignored by the launcher) |
| `observation.state` | 26 | left arm 7, right arm 7, left hand 6, right hand 6 |
| `action` | 26 | same layout, joint targets |
| `task` | string | the language instruction |

Each policy for our six tasks (InsertRing, StackCup, RemoveCap, BuildBlocks, HandoverBanana, DrawerItem) is trained
from 100 demonstrations.

## Rectification

```bash
python scripts/estimate_real_rectification.py --root /path/to/InsertRing --repo-id local/InsertRing --samples 40
```

The script matches SIFT features between the two views and prints the median vertical offset `dy = y_left - y_right`
with its spread, plus disparity statistics. Pass the reported `recommended_rectify_right_dy` to the launcher
(`RECTIFY_RIGHT_DY`). No intrinsics, extrinsics or metric baseline are needed.

## Training

```bash
DATASET_ROOT=/path/to/InsertRing DATASET_REPO_ID=local/InsertRing RUN_NAME=insertring_stereolift \
  RECTIFY_RIGHT_DY=-4.15 bash scripts/train_real_robot.sh
```

Defaults: pi0.5 base, full fine-tuning, 8 GPUs × 8 = global batch 64, 20,000 steps, the shared optimizer settings of
`docs/ROBOCASA365.md`; Fast-FoundationStereo at 320 × 240 (`FFS_SCALE=0.5`, K = 4), assumed field of view 90°,
scale s' = 0.1, M = 16 relation tokens. A smoke test on a few frames:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_policy.py --root /path/to/InsertRing --repo-id local/InsertRing \
    --right-dy -4.15 --scale 0.5 --lift-fov-deg 90 --lift-scale 0.1 --batch 2
```

## Deployment

The policy is a standard LeRobot policy: load it with `PI05StereoLiftPolicy.from_pretrained(path)` (importing
`stereolift` first, or after `scripts/install_plugin.sh`), feed a batch with the two head images, the state and the
instruction through the saved pre-processor, call `select_action`, and un-normalise with the post-processor. Each
50-step chunk is re-planned as in the simulation protocol. After every `predict_action_chunk`,
`policy.last_stereo_confidence` holds the mean stereo confidence c̄ of the observation; thresholding the value of the
first chunk after the approach is the early-stop criterion of the paper.

## Evaluation protocol

Each policy is evaluated over 20 trials per task, with additional perturbation trials: target depth shifts of ±10 cm
and ±20 cm, head tilts of ±5°, and one-second occlusion of one lens during the approach. Stage-wise success (approach,
align, execute) is reported for the three precision tasks InsertRing, StackCup and RemoveCap.
