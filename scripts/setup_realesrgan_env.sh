#!/usr/bin/env bash
set -euo pipefail

# One-click env setup for realesrgan runtime
# Usage:
#   source scripts/setup_realesrgan_env.sh

BASHRC_FILE="${HOME}/.bashrc"

append_if_missing() {
  local line="$1"
  local file="$2"
  if [ ! -f "$file" ]; then
    touch "$file"
  fi
  if ! grep -Fqx "$line" "$file"; then
    echo "$line" >> "$file"
    echo "[added] $line"
  else
    echo "[exists] $line"
  fi
}

# 1) Current session exports
export PATH="/data/dachuang/TEST/miniconda3/bin:${PATH}"
export HOME="/data/dachuang/TEST"
export LD_LIBRARY_PATH="/data/dachuang/envs/realesrgan/lib/python3.8/site-packages/nvidia/nvjitlink/lib:/data/dachuang/envs/realesrgan/lib/python3.8/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

# Keep your last explicit line too
export LD_LIBRARY_PATH="/data/dachuang/envs/realesrgan/lib/python3.8/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

# 2) Persist to ~/.bashrc (idempotent)
append_if_missing 'export HOME=/data/dachuang' "$BASHRC_FILE"
append_if_missing 'export PATH=/data/dachuang/.local/bin:$PATH' "$BASHRC_FILE"
append_if_missing 'export LD_LIBRARY_PATH=/data/dachuang/envs/realesrgan/lib/python3.8/site-packages/nvidia/nvjitlink/lib:/data/dachuang/envs/realesrgan/lib/python3.8/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH' "$BASHRC_FILE"

# 3) Activate env (must be sourced, not executed)
if [ -f "/data/dachuang/envs/realesrgan/bin/activate" ]; then
  # shellcheck disable=SC1091
  source /data/dachuang/envs/realesrgan/bin/activate
  echo "[ok] Activated: /data/dachuang/envs/realesrgan"
else
  echo "[warn] activate not found: /data/dachuang/envs/realesrgan/bin/activate"
fi

echo "[done] Current shell vars applied and ~/.bashrc updated (if needed)."
