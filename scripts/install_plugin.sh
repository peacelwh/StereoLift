#!/usr/bin/env bash
# Optional: make `policy.type=pi05_stereolift` available to plain `lerobot-train` / `lerobot-eval` (and any other
# python process of the environment) without importing stereolift explicitly.  `python -m stereolift.train|eval`
# does not need this.
#
# Writes a .pth file that (1) puts this repository on sys.path and (2) installs a tiny import hook which registers
# the policy right after `lerobot.policies.factory` is imported.  Nothing heavy is imported at start-up.
#
# Usage:  bash scripts/install_plugin.sh [python-binary]
set -euo pipefail
PY=${1:-${PYTHON:-python}}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PTH="${SITE}/stereolift_auto.pth"
{
  echo "${ROOT}"
  echo "import stereolift._sitehook"
} > "${PTH}"
echo "wrote ${PTH}"
"$PY" - <<'EOF'
import sys
print("hook installed:", any(type(f).__name__ == "_StereoLiftFinder" for f in sys.meta_path))
import lerobot.policies.factory as f
print("stereolift registered:", getattr(f, "_stereolift_patched", False))
print("factory resolves class:", f.get_policy_class("pi05_stereolift").__name__)
EOF
