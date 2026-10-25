#!/usr/bin/env bash
# StereoLift training on RoboCasa365 (pi0.5 backbone, the rectified head-mounted agentview stereo pair).
#
# Two protocols of the paper (docs/ROBOCASA365.md):
#   Target50   : Human300 (300 tasks x 100 demos), 75k steps, global batch 64
#   Kitchen-18 : the 18 atomic target tasks x 100 (or 30) demos, global batch 64, cosine schedule over 20k steps
#
#   DATASET_ROOT=/path/to/robocasa365_human300 STATS_OVERRIDE=... TASK_MANIFEST=... RUN_NAME=target50_stereolift \
#       bash scripts/train_robocasa365.sh
#   DATASET_ROOT=/path/to/robocasa365_kitchen18 WARMUP_STEPS=500 STEPS=20000 DECAY_STEPS=20000 \
#       RUN_NAME=kitchen18_100_stereolift bash scripts/train_robocasa365.sh
#
# Every knob is an environment variable; the defaults are the paper's shared recipe (Table of hyper-parameters).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
ACCELERATE=${ACCELERATE:-accelerate}

# ---- data ---------------------------------------------------------------------------------------------------------
DATASET_ROOT=${DATASET_ROOT:?set DATASET_ROOT to a local LeRobot (v3) copy of the RoboCasa365 data}
DATASET_REPO_ID=${DATASET_REPO_ID:-local/robocasa365}
LEFT_KEY=${LEFT_KEY:-observation.images.robot0_agentview_left}
RIGHT_KEY=${RIGHT_KEY:-observation.images.robot0_agentview_right}
# task-balanced normalisation statistics (scripts/compute_task_balanced_stats.py) and the task manifest of the
# RoboCasa365 sampler (scripts/make_task_manifest.py); both default to the files those scripts write
STATS_OVERRIDE=${STATS_OVERRIDE:-${DATASET_ROOT}/meta/stats_task_balanced.json}
TASK_MANIFEST=${TASK_MANIFEST:-${DATASET_ROOT}/meta/task_manifest.parquet}
TASK_SAMPLING_POWER=${TASK_SAMPLING_POWER:-0.4}
EPISODES_FILE=${EPISODES_FILE:-}   # optional episode subset, e.g. configs/kitchen18_100demos.json
RECTIFY_JSON=${RECTIFY_JSON:-${ROOT}/configs/robocasa_agentview_rectification.json}

# ---- backbone and optimisation ---------------------------------------------------------------------------------
PRETRAINED_PATH=${PRETRAINED_PATH:-lerobot/pi05_base}
NUM_PROCESSES=${NUM_PROCESSES:-8}
BATCH_SIZE=${BATCH_SIZE:-8}           # per process; 8 x 8 = the global batch of 64
STEPS=${STEPS:-75000}
DECAY_STEPS=${DECAY_STEPS:-${STEPS}}  # length of the cosine schedule (Kitchen-18: 20000)
WARMUP_STEPS=${WARMUP_STEPS:-300}     # Target50: 300; Kitchen-18: 500
OPTIMIZER_LR=${OPTIMIZER_LR:-5e-5}
DECAY_LR=${DECAY_LR:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
GRAD_CLIP=${GRAD_CLIP:-1.0}
ADAPTER_LR_SCALE=${ADAPTER_LR_SCALE:-10}
SAVE_FREQ=${SAVE_FREQ:-5000}
LOG_FREQ=${LOG_FREQ:-50}
SEED=${SEED:-1000}

# ---- StereoLift interface (256 x 256 renders) --------------------------------------------------------------------
FFS_SCALE=${FFS_SCALE:-1.0}
FFS_ITERS=${FFS_ITERS:-4}
LIFT_FOV_DEG=${LIFT_FOV_DEG:-60}
LIFT_SCALE=${LIFT_SCALE:-0.03}
FOURIER_MAX_LOG2=${FOURIER_MAX_LOG2:-4}
NUM_RELATION_TOKENS=${NUM_RELATION_TOKENS:-16}        # ablation B: 0
USE_GEOMETRIC_EMBEDDING=${USE_GEOMETRIC_EMBEDDING:-true}  # ablation C: false
BELIEF_POINT_ONLY=${BELIEF_POINT_ONLY:-false}         # ablation D: true
RELATION_SEMANTIC_KEYS=${RELATION_SEMANTIC_KEYS:-true}

# ---- bookkeeping -------------------------------------------------------------------------------------------------
RUN_NAME=${RUN_NAME:-stereolift_robocasa365}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/outputs/${RUN_NAME}}
LOG_FILE=${LOG_FILE:-${ROOT}/outputs/logs/${RUN_NAME}.log}
mkdir -p "$(dirname "${LOG_FILE}")"

[[ -f "${RECTIFY_JSON}" ]] || { echo "[error] missing ${RECTIFY_JSON} (see scripts/compute_robocasa_rectification.py)" >&2; exit 2; }
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || { echo "[error] dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

ACCELERATE_ARGS=(--num_processes="${NUM_PROCESSES}" --num_machines=1 --main_process_port="${MAIN_PROCESS_PORT:-29715}" --mixed_precision=bf16 --dynamo_backend=no)
if (( NUM_PROCESSES > 1 )); then ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}"); fi

EXTRA_ARGS=()
if [[ -f "${STATS_OVERRIDE}" ]]; then EXTRA_ARGS+=(--dataset.stats_override_path="${STATS_OVERRIDE}"); else echo "[warning] no stats override at ${STATS_OVERRIDE}; using the dataset's meta/stats.json" >&2; fi
if [[ -f "${TASK_MANIFEST}" ]]; then
  EXTRA_ARGS+=(--dataset.task_sampling_manifest="${TASK_MANIFEST}" --dataset.task_sampling_power="${TASK_SAMPLING_POWER}")
else
  echo "[warning] no task manifest at ${TASK_MANIFEST}; sampling frames uniformly" >&2
fi
if [[ -n "${EPISODES_FILE}" ]]; then
  EPISODES=$("${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["episodes"]))' "${EPISODES_FILE}")
  EXTRA_ARGS+=(--dataset.episodes="${EPISODES}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume=true --config_path="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json")
fi

echo "StereoLift | dataset ${DATASET_ROOT} | init ${PRETRAINED_PATH} | ${NUM_PROCESSES} x ${BATCH_SIZE} | steps ${STEPS} (cosine over ${DECAY_STEPS}) | output ${OUTPUT_DIR}"
cd "${ROOT}"
"${ACCELERATE}" launch "${ACCELERATE_ARGS[@]}" -m stereolift.train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend="${VIDEO_BACKEND:-pyav}" \
  --selected_image_features="[\"${LEFT_KEY}\",\"${RIGHT_KEY}\"]" \
  --policy.type=pi05_stereolift \
  --policy.pretrained_path="${PRETRAINED_PATH}" \
  --policy.device=cuda \
  --policy.dtype="${POLICY_DTYPE:-bfloat16}" \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.optimizer_lr="${OPTIMIZER_LR}" \
  --policy.optimizer_weight_decay="${WEIGHT_DECAY}" \
  --policy.optimizer_grad_clip_norm="${GRAD_CLIP}" \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.scheduler_decay_steps="${DECAY_STEPS}" \
  --policy.scheduler_decay_lr="${DECAY_LR}" \
  --policy.stereo_left_key="${LEFT_KEY}" \
  --policy.stereo_right_key="${RIGHT_KEY}" \
  --policy.rectify_mode=homography \
  --policy.rectify_homography_path="${RECTIFY_JSON}" \
  --policy.ffs_input_scale="${FFS_SCALE}" \
  --policy.ffs_iters="${FFS_ITERS}" \
  --policy.lift_fov_deg="${LIFT_FOV_DEG}" \
  --policy.lift_scale="${LIFT_SCALE}" \
  --policy.lift_fourier_max_log2="${FOURIER_MAX_LOG2}" \
  --policy.num_relation_tokens="${NUM_RELATION_TOKENS}" \
  --policy.use_geometric_embedding="${USE_GEOMETRIC_EMBEDDING}" \
  --policy.belief_point_only="${BELIEF_POINT_ONLY}" \
  --policy.relation_semantic_keys="${RELATION_SEMANTIC_KEYS}" \
  --policy.adapter_lr_scale="${ADAPTER_LR_SCALE}" \
  --policy.push_to_hub=false \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS:-4}" \
  --steps="${STEPS}" \
  --log_freq="${LOG_FREQ}" \
  --save_freq="${SAVE_FREQ}" \
  --seed="${SEED}" \
  --env_eval_freq=0 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --wandb.enable="${WANDB_ENABLE:-false}" \
  --wandb.project="${WANDB_PROJECT:-stereolift}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
