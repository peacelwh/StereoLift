# RoboCasa365: Target50 (Table 1) and Kitchen-18 (Table 2)

Both protocols use the head-mounted stereo pair `robot0_agentview_left/right` of RoboCasa365 (256 × 256 pixels), the
fixed rectifying homographies in `configs/robocasa_agentview_rectification.json`, the pi0.5 backbone and the shared
training recipe below. The right view enters the policy only through the frozen stereo model, never as a second set
of visual tokens.

## Relevant files

Training
* `scripts/train_robocasa365.sh`: launcher (accelerate + `python -m stereolift.train`)
* `scripts/make_task_manifest.py`: task blocks of a RoboCasa365 dataset for the task-balanced sampler
* `scripts/compute_task_balanced_stats.py`: task-balanced normalisation statistics of state and action
* `configs/kitchen18_tasks.json`, `configs/kitchen18_{100,30}demos.json`: the Kitchen-18 protocol
  (`scripts/make_kitchen18_subsets.py` regenerates them)

Evaluation
* `scripts/eval_robocasa365.sh`: sharded evaluation over the Target50 groups or a task list, with summary
* `scripts/eval_robocasa_multitask.py`, `scripts/eval_robocasa_task.sh`: one process per shard, one `lerobot-eval` run per task
* `stereolift/robocasa_perturb.py`: test-time rig perturbations (robustness study)

## Setup

1. Install the repository (`scripts/setup_third_party.sh`, see the README).
2. Install RoboCasa365 for evaluation following the official instructions (https://robocasa.ai): `robosuite`,
   `robocasa` with the RoboCasa365 kitchens, objects (`objaverse`, `lightwheel`) and the task registry. We used
   robocasa 1.0.1, robosuite 1.5.2 and MuJoCo 3.3.1. Rendering is headless (`MUJOCO_GL=egl`).
3. Download the pi0.5 base checkpoint `lerobot/pi05_base` (done automatically by lerobot on first use).

## Data

**Human300 (Target50 protocol).** The 300-task, 30,000-demonstration pretraining set in LeRobot v3 format. We used
the LeRobot conversion `Whalswp/robocasa_pretrain_human300_v3`; the official release is at
`ember-lab-berkeley/robocasa365-pretrain-*`. Only the two agentview cameras are needed. The episodes of task *i*
are the block `100 i .. 100 i + 99` (tasks ordered alphabetically inside each family, 65 atomic then 235
composite).

```bash
DATA=/path/to/robocasa365_human300
python scripts/make_task_manifest.py --dataset-root $DATA --episodes-per-task 100    # meta/task_manifest.parquet
python scripts/compute_task_balanced_stats.py --dataset-root $DATA                   # meta/stats_task_balanced.json
```

**Kitchen-18.** The 18 atomic target tasks of Target50 (`configs/kitchen18_tasks.json`) with 100 or 30 human
demonstrations per task, drawn from Human300. We release them as a stand-alone LeRobot dataset with the two
agentview cameras, the task manifest and the task-balanced statistics:

```bash
hf download Bluexxx/robocasa365_kitchen18 --repo-type dataset --local-dir /path/to/robocasa365_kitchen18
```

Episode `100 k .. 100 k + 99` of the released dataset is task *k* of `configs/kitchen18_tasks.json`. The
100-demonstration protocol uses all 1,800 episodes; the 30-demonstration protocol uses
`configs/kitchen18_30demos_release.json` (the first 30 episodes of every task) with
`meta/stats_task_balanced_30demos.json`. If you build the subsets from the full Human300 copy instead, pass
`configs/kitchen18_100demos.json` or `configs/kitchen18_30demos.json` (Human300 episode indices).

**Rectification.** `configs/robocasa_agentview_rectification.json` holds the homographies of the toe-in pair
(baseline 0.70 m, inward rotation 23.4°, rectified focal length 221.7 pixels), computed once from the simulator's
camera parameters with `scripts/compute_robocasa_rectification.py`. They are applied inside the policy; the policy
itself never receives camera parameters.

## Shared training recipe

| Hyper-parameter | Value |
|---|---|
| Backbone and expert | pi0.5 (PaliGemma 3B and a 300M flow-matching expert), bf16, gradient checkpointing |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.01, gradient clipping at norm 1.0 |
| Learning rate | 5e-5 peak, cosine decay to 5e-6, 300 (Target50) or 500 (Kitchen-18) warm-up steps; 10× multiplier for the added interface parameters; assumed intrinsics fixed |
| Batch and steps | 64 / 75k (Target50); 64, cosine schedule defined over 20k steps (Kitchen-18, 100 and 30 demonstrations) |
| Action chunk | 50 steps |
| Fast-FoundationStereo | frozen; input scale 1.0; K = 4; d_max = 192 pixels |
| Assumed intrinsics | field of view 60°, principal point at the image centre, scale s' = 0.03; fixed |
| Embedding | width 256, 4 heads; 8 fixed log-spaced Fourier bands per axis up to 2^4 π/2; zero-initialised output projection |
| Relation tokens | M = 16, one cross-attention layer over all cells, language and state FiLM, semantic key term on, key penalty λ = 4 |
| Loss | flow matching only |
| Normalization | task-balanced mean and standard deviation for state and action (RoboCasa365 recipe) |

These are the defaults of `scripts/train_robocasa365.sh`. Every value is an environment variable of the launcher.

## Table 1: Target50 (official protocol)

Train on Human300 for 75,000 steps at a global batch size of 64 (8 GPUs × 8), sampling tasks with the RoboCasa365
task-balanced sampler:

```bash
DATASET_ROOT=/path/to/robocasa365_human300 RUN_NAME=target50_stereolift \
  bash scripts/train_robocasa365.sh
```

Evaluate the 75k checkpoint on the 50 target tasks in the pretraining kitchens (`pretrain` split, official object
registries), 50 rollouts per task, seed 7 with per-episode offsets, the official per-task horizon and re-planning
every 25 steps of the 50-step action chunk. The three groups are atomic seen (18 tasks), composite seen (16) and
composite unseen (16); the overall number is the task-weighted mean.

```bash
CHECKPOINT=outputs/target50_stereolift/checkpoints/075000/pretrained_model \
OUTPUT_ROOT=outputs/eval/target50_stereolift \
EVAL_GROUPS="atomic_seen composite_seen composite_unseen" N_EPISODES=50 N_ACTION_STEPS=25 \
GPU_IDS=0,1,2,3,4,5,6,7 SHARDS_PER_GPU=2 bash scripts/eval_robocasa365.sh
```

`summary.json` reports the per-group task means and the overall mean; `per_task.csv` lists every task.

## Table 2: Kitchen-18

Joint training on the 18 atomic target tasks with 100 or 30 human demonstrations per task, global batch 64, a
cosine learning-rate schedule defined over 20,000 steps and 500 warm-up steps. Checkpoints are written every 5,000
steps.

```bash
# 100 demonstrations per task
DATASET_ROOT=/path/to/robocasa365_kitchen18 RUN_NAME=kitchen18_100_stereolift \
  WARMUP_STEPS=500 STEPS=20000 DECAY_STEPS=20000 SAVE_FREQ=5000 bash scripts/train_robocasa365.sh

# 30 demonstrations per task
DATASET_ROOT=/path/to/robocasa365_kitchen18 RUN_NAME=kitchen18_30_stereolift \
  EPISODES_FILE=configs/kitchen18_30demos_release.json STATS_OVERRIDE=/path/to/robocasa365_kitchen18/meta/stats_task_balanced_30demos.json \
  WARMUP_STEPS=500 STEPS=20000 DECAY_STEPS=20000 SAVE_FREQ=5000 bash scripts/train_robocasa365.sh
```

Evaluation: 50 rollouts per task, re-planning every 5 steps, the same scene split, registries, seed and horizons as
above. Report the mean success over the 18 task means, its standard error, the mean over the five pick-and-place
tasks (`PickPlace*`) and paired per-task differences against the comparison policy.

```bash
CHECKPOINT=outputs/kitchen18_100_stereolift/checkpoints/<step>/pretrained_model \
OUTPUT_ROOT=outputs/eval/kitchen18_100_stereolift \
TASK_FILE=configs/kitchen18_tasks.json N_EPISODES=50 N_ACTION_STEPS=5 GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/eval_robocasa365.sh
```

### Ablations (Table 3, 30 demonstrations per task)

| ID | Variant | Launcher setting |
|---|---|---|
| A | full model | defaults |
| B | without relation tokens | `NUM_RELATION_TOKENS=0` |
| C | without geometric token embedding | `USE_GEOMETRIC_EMBEDDING=false` |
| D | point estimate instead of belief ([b; c] → d_K) | `BELIEF_POINT_ONLY=true` |

### Robustness, baseline scaling, assumed intrinsics and anytime refinement (Section Q3)

All of these evaluate the nominal Kitchen-18 policy without retraining, on the six tasks of
`configs/kitchen6_tasks.json` with 20 rollouts per task:

```bash
COMMON="CHECKPOINT=... TASK_FILE=configs/kitchen6_tasks.json N_EPISODES=20 N_ACTION_STEPS=5"
# rigid motions of the stereo pair (yaw / pitch in degrees, translations in metres)
env $COMMON OUTPUT_ROOT=outputs/eval/k6_yaw+10 CAM_PERTURB='{"yaw_deg":10}' bash scripts/eval_robocasa365.sh
env $COMMON OUTPUT_ROOT=outputs/eval/k6_dz+5cm CAM_PERTURB='{"dz":0.05}' bash scripts/eval_robocasa365.sh
# baseline scaling about the rig midpoint (B/Z 0.15 ... 1.05 = scale 0.21 ... 1.5 of the nominal 0.70)
env $COMMON OUTPUT_ROOT=outputs/eval/k6_bs0.5 CAM_PERTURB='{"baseline_scale":0.5}' bash scripts/eval_robocasa365.sh
# assumed field of view at test time, and the number of refinement iterations K
env $COMMON OUTPUT_ROOT=outputs/eval/k6_fov45 POLICY_ARGS="--policy.lift_fov_deg=45" bash scripts/eval_robocasa365.sh
env $COMMON OUTPUT_ROOT=outputs/eval/k6_K0 POLICY_ARGS="--policy.ffs_iters=0" bash scripts/eval_robocasa365.sh
```

## Notes

* `eval_robocasa_task.sh` derives the episode horizon from RoboCasa's task registry and the camera list from the
  checkpoint's `input_features`.
* Headless rendering: if your container's glvnd has no NVIDIA vendor ICD, MuJoCo silently falls back to software
  rendering (~3 environment steps per second). `STEREOLIFT_NVIDIA_EGL=1` points glvnd at
  `configs/10_nvidia_egl.json`; `STEREOLIFT_EGL_SINGLE_DEVICE=1` handles containers that enumerate a single EGL device.
* Evaluation is resumable: tasks with a complete `eval_info.json` are skipped; `FORCE=1` re-runs them.
* `RECORDING=1` saves one video per rollout.
