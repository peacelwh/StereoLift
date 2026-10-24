#!/usr/bin/env bash
# Install the two third-party components StereoLift builds on, into third_party/:
#   1. LeRobot at the pinned commit + the small patch in third_party/lerobot_stereolift.patch (editable install)
#   2. Fast-FoundationStereo (code) + the released c-ffs weights from the Hugging Face Hub
#
# Usage (inside the conda env that already has torch):   bash scripts/setup_third_party.sh
# Environment: LEROBOT_COMMIT (default e40b58a8), HF_ENDPOINT (e.g. https://hf-mirror.com behind a firewall),
#              SKIP_LEROBOT=1 / SKIP_FFS=1 to run only one half.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TP="${ROOT}/third_party"
PY=${PYTHON:-python}
LEROBOT_COMMIT=${LEROBOT_COMMIT:-e40b58a8dfa9e7b86918c374791599d070518d11}
mkdir -p "${TP}"

if [[ "${SKIP_LEROBOT:-0}" != "1" ]]; then
  if [[ ! -d "${TP}/lerobot/.git" ]]; then
    git clone https://github.com/huggingface/lerobot.git "${TP}/lerobot"
  fi
  git -C "${TP}/lerobot" fetch --depth=1 origin "${LEROBOT_COMMIT}" || git -C "${TP}/lerobot" fetch origin
  git -C "${TP}/lerobot" checkout -q "${LEROBOT_COMMIT}"
  if git -C "${TP}/lerobot" apply --check "${TP}/lerobot_stereolift.patch" 2>/dev/null; then
    git -C "${TP}/lerobot" apply "${TP}/lerobot_stereolift.patch"
    echo "applied lerobot_stereolift.patch"
  elif git -C "${TP}/lerobot" apply --reverse --check "${TP}/lerobot_stereolift.patch" 2>/dev/null; then
    echo "lerobot_stereolift.patch already applied"
  else
    echo "[error] lerobot_stereolift.patch does not apply cleanly to ${LEROBOT_COMMIT}" >&2
    exit 1
  fi
  # pi0.5 (PaliGemma + action expert) and the RoboCasa365 benchmark integration
  "${PY}" -m pip install -e "${TP}/lerobot[pi]"
fi

if [[ "${SKIP_FFS:-0}" != "1" ]]; then
  if [[ ! -d "${TP}/Fast-FoundationStereo/core" ]]; then
    git clone https://github.com/NVlabs/Fast-FoundationStereo.git "${TP}/Fast-FoundationStereo"
  fi
  "${PY}" -m pip install timm einops omegaconf scipy scikit-image imageio pyyaml
  "${PY}" - <<EOF
from pathlib import Path
from huggingface_hub import hf_hub_download
import hashlib
dst = Path("${TP}/Fast-FoundationStereo/weights/c-ffs")
dst.mkdir(parents=True, exist_ok=True)
for name in ("model_best_bp2_serialize.pth", "cfg.yaml"):
    p = hf_hub_download("nvidia/c-fast-foundationstereo", name, local_dir=str(dst))
    print("downloaded", p)
md5 = hashlib.md5((dst / "model_best_bp2_serialize.pth").read_bytes()).hexdigest()
print("c-ffs md5", md5, "(expected 37459c57fd00c61bedf409f27ee10756)")
EOF
fi

"${PY}" -m pip install -e "${ROOT}"
echo "done. Try:  ${PY} -m pytest ${ROOT}/tests -q"
