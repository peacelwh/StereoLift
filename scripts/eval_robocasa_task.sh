#!/usr/bin/env bash
# Evaluate one checkpoint on one RoboCasa365 task with lerobot-eval (run inside the RoboCasa environment).
#
# Required: CHECKPOINT (a pretrained_model directory), TASK_NAME.
# Optional: SPLIT (pretrain = the official scene split), N_EPISODES (50), SEED (7), N_ACTION_STEPS (re-planning
#           interval: 5 for Kitchen-18, 25 for Target50), OBJ_REGISTRIES, OUTPUT_DIR, RUN_NAME, EPISODE_LENGTH
#           (default: the official per-task horizon), EXTRA_POLICY_ARGS (e.g. "--policy.ffs_iters=0
#           --policy.lift_fov_deg=45"), STEREOLIFT_CAM_PERTURB (JSON rig perturbation, see
#           stereolift/robocasa_perturb.py), CUDA_VISIBLE_DEVICES (one GPU), MUJOCO_EGL_DEVICE_ID.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
CHECKPOINT=${CHECKPOINT:-}
TASK_NAME=${TASK_NAME:-CloseFridge}
SPLIT=${SPLIT:-pretrain}
OBJ_REGISTRIES=${OBJ_REGISTRIES:-'["objaverse","lightwheel"]'}

[[ -n "${CHECKPOINT}" && -f "${CHECKPOINT}/config.json" ]] || { echo "[error] CHECKPOINT must contain config.json" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Headless GPU rendering.  If glvnd has no NVIDIA vendor ICD in your container, MuJoCo silently falls back to
# software rendering (~3 env steps/s); pointing glvnd at configs/10_nvidia_egl.json restores NVIDIA EGL.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}
if [[ -n "${STEREOLIFT_NVIDIA_EGL:-}" ]]; then export __EGL_VENDOR_LIBRARY_FILENAMES=${ROOT}/configs/10_nvidia_egl.json; fi
export TOKENIZERS_PARALLELISM=false
# cap CPU threads: with many shards on one machine, torch's default (one thread per core) oversubscribes the CPUs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export PYTHONPATH=${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

# the cameras the checkpoint was trained on, in its input_features order
POLICY_CAMERAS=$("${PYTHON_BIN}" - "${CHECKPOINT}/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
t = c.get("type") or c.get("policy_type")
if t != "pi05_stereolift":
    raise SystemExit(f"unsupported policy type {t!r}")
cams = [k.split("observation.images.", 1)[1] for k in c.get("input_features", {}) if k.startswith("observation.images.")]
assert {"robot0_agentview_left", "robot0_agentview_right"} <= set(cams), cams
print(",".join(cams))
PY
)
CAMERA_NAME=${CAMERA_NAME:-${POLICY_CAMERAS}}

if [[ -z "${EPISODE_LENGTH:-}" ]]; then
  EPISODE_LENGTH=$(TASK_NAME="${TASK_NAME}" "${PYTHON_BIN}" - <<'PY' 2>/dev/null | tail -n 1
import os
from robocasa.utils.dataset_registry_utils import get_task_horizon
print(int(get_task_horizon(os.environ["TASK_NAME"])))
PY
  )
fi
[[ "${EPISODE_LENGTH}" =~ ^[1-9][0-9]*$ ]] || { echo "[error] bad horizon for ${TASK_NAME}: ${EPISODE_LENGTH:-<empty>}" >&2; exit 2; }

RUN_NAME=${RUN_NAME:-robocasa_${TASK_NAME}_${SPLIT}_eval}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/outputs/eval/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "${OUTPUT_DIR}"
echo "task ${TASK_NAME} (${SPLIT}, horizon ${EPISODE_LENGTH}) | cameras ${CAMERA_NAME} | episodes ${N_EPISODES:-50} | seed ${SEED:-7} | replan ${N_ACTION_STEPS:-5}"
echo "checkpoint ${CHECKPOINT} | extra policy args: ${EXTRA_POLICY_ARGS:-<none>} | rig perturbation: ${STEREOLIFT_CAM_PERTURB:-<none>}"
echo "output ${OUTPUT_DIR}"

cd "${ROOT}"
# shellcheck disable=SC2086
"${PYTHON_BIN}" -m stereolift.eval \
  --policy.path="${CHECKPOINT}" \
  --policy.n_action_steps="${N_ACTION_STEPS:-5}" \
  ${EXTRA_POLICY_ARGS:-} \
  --env.type=robocasa \
  --env.task="${TASK_NAME}" \
  --env.split="${SPLIT}" \
  --env.episode_length="${EPISODE_LENGTH}" \
  --env.camera_name="${CAMERA_NAME}" \
  --env.observation_height="${OBSERVATION_HEIGHT:-256}" \
  --env.observation_width="${OBSERVATION_WIDTH:-256}" \
  --env.obj_registries="${OBJ_REGISTRIES}" \
  --eval.n_episodes="${N_EPISODES:-50}" \
  --eval.batch_size="${EVAL_BATCH_SIZE:-1}" \
  --eval.use_async_envs=false \
  --eval.recording="${RECORDING:-false}" \
  --seed="${SEED:-7}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}"
