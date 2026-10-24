#!/usr/bin/env bash
# Sharded RoboCasa365 evaluation of one checkpoint over the Target50 groups or a task list, then a summary.
#
#   # Table 1: official Target50 protocol (50 tasks, 50 rollouts, re-planning every 25 steps)
#   CHECKPOINT=outputs/target50_stereolift/checkpoints/075000/pretrained_model OUTPUT_ROOT=outputs/eval/target50_stereolift \
#     EVAL_GROUPS="atomic_seen composite_seen composite_unseen" N_ACTION_STEPS=25 GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/eval_robocasa365.sh
#   # Table 2: Kitchen-18 (18 tasks, 50 rollouts, re-planning every 5 steps)
#   CHECKPOINT=... OUTPUT_ROOT=... TASK_FILE=configs/kitchen18_tasks.json N_ACTION_STEPS=5 bash scripts/eval_robocasa365.sh
#   # robustness: CAM_PERTURB='{"yaw_deg":10}' TASK_FILE=configs/kitchen6_tasks.json N_EPISODES=20 ...
#   # anytime / assumed intrinsics: POLICY_ARGS="--policy.ffs_iters=0" or POLICY_ARGS="--policy.lift_fov_deg=45"
#
# Results: <OUTPUT_ROOT>/<group>/<idx>_<task>/eval_info.json, summary.json, per_task.csv.  Completed tasks are
# skipped on re-runs (resumable).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT}
GPU_IDS=${GPU_IDS:-0,1,2,3}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-2}
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_SHARDS=$(( ${#GPUS[@]} * SHARDS_PER_GPU ))

SEL=()
if [[ -n "${EVAL_GROUPS:-}" ]]; then read -r -a G <<< "${EVAL_GROUPS}"; SEL+=(--groups "${G[@]}"); fi
if [[ -n "${TASK_FILE:-}" ]]; then SEL+=(--task-file "${TASK_FILE}"); fi
if [[ -n "${TASKS:-}" ]]; then read -r -a T <<< "${TASKS}"; SEL+=(--tasks "${T[@]}"); fi
COMMON=(--checkpoint "${CHECKPOINT}" --output-root "${OUTPUT_ROOT}" --episodes-per-task "${N_EPISODES:-50}"
        --seed "${SEED:-7}" --n-action-steps "${N_ACTION_STEPS:-5}" --split "${SPLIT:-pretrain}"
        --object-registries "${OBJ_REGISTRIES:-[\"objaverse\",\"lightwheel\"]}" "${SEL[@]}")
[[ -n "${POLICY_ARGS:-}" ]] && COMMON+=("--policy-args=${POLICY_ARGS}")
[[ -n "${CAM_PERTURB:-}" ]] && COMMON+=("--cam-perturb=${CAM_PERTURB}")
[[ "${RECORDING:-0}" == "1" ]] && COMMON+=(--recording)
[[ "${FORCE:-0}" == "1" ]] && COMMON+=(--force)

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${ROOT}"
echo "checkpoint ${CHECKPOINT}"; echo "output ${OUTPUT_ROOT}"; echo "${NUM_SHARDS} shards on GPUs ${GPU_IDS}"
pids=()
for (( s=0; s<NUM_SHARDS; s++ )); do
  gpu=${GPUS[$(( s % ${#GPUS[@]} ))]}
  "${PYTHON_BIN}" scripts/eval_robocasa_multitask.py "${COMMON[@]}" --num-shards "${NUM_SHARDS}" --shard-index "${s}" --gpu "${gpu}" \
     > "${OUTPUT_ROOT}/logs/shard_${s}.log" 2>&1 &
  pids+=("$!")
  sleep "${STAGGER_S:-20}"
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
"${PYTHON_BIN}" scripts/eval_robocasa_multitask.py "${COMMON[@]}" --summarize-only || true
exit "${status}"
