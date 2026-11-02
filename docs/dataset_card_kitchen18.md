---
license: cc-by-nc-sa-4.0
task_categories:
- robotics
tags:
- LeRobot
- robocasa365
- stereo
- manipulation
- StereoLift
pretty_name: RoboCasa365 Kitchen-18 (stereo)
size_categories:
- 100K<n<1M
---

# RoboCasa365 Kitchen-18 (stereo agentview pair)

The **Kitchen-18** protocol of the paper *StereoLift: Enhancing Vision-Language-Action Models for Precise
Manipulation via Calibration-Free Stereo Lifting*: the 18 atomic target tasks of the RoboCasa365 Target50 benchmark
with 100 human demonstrations per task, drawn from the RoboCasa365 Human300 pretraining set, in
[LeRobot](https://github.com/huggingface/lerobot) v3 format with the head-mounted stereo pair
`robot0_agentview_left` / `robot0_agentview_right` (256 × 256 pixels, 20 Hz).

Code, rectification files and launch scripts: https://github.com/peacelwh/StereoLift (see `docs/ROBOCASA365.md`).

## Contents

| | |
|---|---|
| Episodes | 1,800 (18 tasks × 100 human demonstrations) |
| Frames | 406,831 at 20 Hz |
| Cameras | `observation.images.robot0_agentview_left`, `observation.images.robot0_agentview_right` (256 × 256 × 3, H.264) |
| State | `observation.state`, 16-d (base pose, base-relative end-effector pose, gripper) |
| Action | `action`, 12-d (base motion, control mode, end-effector delta, gripper) |
| Language | `task` (the instruction), `annotation.human.task_description`, `annotation.human.task_name` |
| Robot | PandaOmron (mobile Franka arm with a parallel gripper) |

Episodes `100 k .. 100 k + 99` belong to task `k` of the list below (the order of `configs/kitchen18_tasks.json` in
the code repository):

```
 0 CloseBlenderLid            6 OpenDrawer                 12 PickPlaceToasterToCounter
 1 CloseFridge                7 OpenStandMixerHead         13 SlideDishwasherRack
 2 CloseToasterOvenDoor       8 PickPlaceCounterToCabinet  14 TurnOffStove
 3 CoffeeSetupMug             9 PickPlaceCounterToStove    15 TurnOnElectricKettle
 4 NavigateKitchen           10 PickPlaceDrawerToCounter   16 TurnOnMicrowave
 5 OpenCabinet               11 PickPlaceSinkToCounter     17 TurnOnSinkFaucet
```

Extra files:

* `meta/task_manifest.parquet`: the 18 task blocks, for the RoboCasa365 task-balanced sampler
  (`--dataset.task_sampling_manifest`).
* `meta/stats_task_balanced.json`: task-balanced mean and standard deviation of state and action (every task weighs
  the same), the normalisation used in the paper (`--dataset.stats_override_path`).
* `kitchen18_30demos.json` and `meta/stats_task_balanced_30demos.json`: the 30-demonstration protocol (the first 30
  episodes of every task) and its statistics.

## Protocol

Policies are trained jointly on the 18 tasks (global batch 64, cosine learning-rate schedule defined over 20,000
steps) and evaluated with 50 rollouts per task in the official `pretrain` scene split with the official object
registries, seed 7 with per-episode offsets, the official per-task horizon and re-planning every 5 steps of a 50-step
action chunk; results are reported as the mean success over the 18 task means with its standard error, the mean over
the five pick-and-place tasks and paired per-task differences.

## How it was built

Extracted from the LeRobot conversion of Human300 (`Whalswp/robocasa_pretrain_human300_v3`; official release at
`ember-lab-berkeley/robocasa365-pretrain-*`) with `lerobot.datasets.dataset_tools.split_dataset`, keeping the episodes
`100 i .. 100 i + 99` of the 18 tasks (Human300 orders its 300 tasks alphabetically inside each family, 65 atomic
then 235 composite), then dropping the `robot0_eye_in_hand` camera with `remove_feature`. Frames, states, actions
and instructions are unchanged.

## Loading

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("Bluexxx/robocasa365_kitchen18")
print(ds.meta.total_episodes, ds.meta.camera_keys)
```

## License and attribution

The demonstrations are part of RoboCasa365 (https://robocasa.ai) and keep its terms; this redistribution is for
non-commercial research. Please cite RoboCasa365 and, if you use this subset or the code, StereoLift:

```bibtex
@misc{li2026stereolift,
  title  = {StereoLift: Enhancing Vision-Language-Action Models for Precise Manipulation via Calibration-Free Stereo Lifting},
  author = {Li, Wenhao and Liang, Xusheng and Shi, Yang and Han, Zhongyi and Qin, Zheyun and Yin, Yilong},
  year   = {2026},
  howpublished = {\url{https://github.com/peacelwh/StereoLift}}
}
```
