#!/usr/bin/env bash
# StereoLift fine-tuning on a real-robot LeRobot dataset (the paper's Unitree G1-D with two BrainCo hands:
# 640 x 480 head stereo pair, 26-d state and action).  See docs/REAL_ROBOT.md.
#
#   DATASET_ROOT=/data/InsertRing DATASET_REPO_ID=local/InsertRing RUN_NAME=insertring_stereolift bash scripts/train_real_robot.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
ACCELERATE=${ACCELERATE:-accelerate}

DATASET_ROOT=${DATASET_ROOT:?set DATASET_ROOT to the LeRobot dataset of the task}
DATASET_REPO_ID=${DATASET_REPO_ID:-local/real_robot_task}
LEFT_KEY=${LEFT_KEY:-observation.images.head_left}
RIGHT_KEY=${RIGHT_KEY:-observation.images.head_right}
PRETRAINED_PATH=${PRETRAINED_PATH:-lerobot/pi05_base}

# the only calibration correction: the constant vertical offset of the right image, estimated from the data with
# scripts/estimate_real_rectification.py (-4.15 px for the paper's head pair)
RECTIFY_RIGHT_DY=${RECTIFY_RIGHT_DY:--4.15}
FFS_SCALE=${FFS_SCALE:-0.5}        # 640 x 480 -> 320 x 240
FFS_ITERS=${FFS_ITERS:-4}
LIFT_FOV_DEG=${LIFT_FOV_DEG:-90}
LIFT_SCALE=${LIFT_SCALE:-0.1}
FOURIER_MAX_LOG2=${FOURIER_MAX_LOG2:-6}
NUM_RELATION_TOKENS=${NUM_RELATION_TOKENS:-16}

NUM_PROCESSES=${NUM_PROCESSES:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
STEPS=${STEPS:-20000}
DECAY_STEPS=${DECAY_STEPS:-${STEPS}}
WARMUP_STEPS=${WARMUP_STEPS:-500}
OPTIMIZER_LR=${OPTIMIZER_LR:-5e-5}
DECAY_LR=${DECAY_LR:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
GRAD_CLIP=${GRAD_CLIP:-1.0}
ADAPTER_LR_SCALE=${ADAPTER_LR_SCALE:-10}
SAVE_FREQ=${SAVE_FREQ:-5000}
SEED=${SEED:-1000}

RUN_NAME=${RUN_NAME:-stereolift_real}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/outputs/${RUN_NAME}}
LOG_FILE=${LOG_FILE:-${ROOT}/outputs/logs/${RUN_NAME}.log}
mkdir -p "$(dirname "${LOG_FILE}")"
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || { echo "[error] dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

ACCELERATE_ARGS=(--num_processes="${NUM_PROCESSES}" --num_machines=1 --main_process_port="${MAIN_PROCESS_PORT:-29716}" --mixed_precision=bf16 --dynamo_backend=no)
if (( NUM_PROCESSES > 1 )); then ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}"); fi
EXTRA_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume=true --config_path="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json")
fi

echo "StereoLift (real robot) | dataset ${DATASET_ROOT} | init ${PRETRAINED_PATH} | ${NUM_PROCESSES} x ${BATCH_SIZE} | steps ${STEPS} | output ${OUTPUT_DIR}"
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
  --policy.rectify_mode=vertical_shift \
  --policy.rectify_right_dy="${RECTIFY_RIGHT_DY}" \
  --policy.ffs_input_scale="${FFS_SCALE}" \
  --policy.ffs_iters="${FFS_ITERS}" \
  --policy.lift_fov_deg="${LIFT_FOV_DEG}" \
  --policy.lift_scale="${LIFT_SCALE}" \
  --policy.lift_fourier_max_log2="${FOURIER_MAX_LOG2}" \
  --policy.num_relation_tokens="${NUM_RELATION_TOKENS}" \
  --policy.adapter_lr_scale="${ADAPTER_LR_SCALE}" \
  --policy.push_to_hub=false \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS:-4}" \
  --steps="${STEPS}" \
  --log_freq="${LOG_FREQ:-50}" \
  --save_freq="${SAVE_FREQ}" \
  --seed="${SEED}" \
  --env_eval_freq=0 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --wandb.enable="${WANDB_ENABLE:-false}" \
  --wandb.project="${WANDB_PROJECT:-stereolift}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
