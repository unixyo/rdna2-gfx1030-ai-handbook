#!/usr/bin/env bash
# Starts this project's pinned ComfyUI; never touches Qwen's ComfyUI.
# Requires the separate MOSS venv and already installed dependencies.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
port="${MOSS_PORT:-8299}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
  echo "Invalid MOSS_PORT: $port" >&2
  exit 2
fi
if [[ ! -x "$root/.venv-moss/bin/python" ]]; then
  echo 'Missing independent MOSS venv' >&2
  exit 3
fi
if [[ "$(git -C "$root/runtime/ComfyUI" rev-parse HEAD)" != '086099f5d8886163ef77dc354f8382cb7431582a' ]]; then
  echo 'MOSS ComfyUI source revision mismatch' >&2
  exit 4
fi
# This project's gfx1030 INT8 models require the local Triton replacement:
# hipBLASLt has no TensileLibrary_lazy_gfx1030.dat. Set =0 only for A/B debugging.
export MOSS_RDNA2_INT8_PATCH="${MOSS_RDNA2_INT8_PATCH:-1}"
mkdir -p -- "$root/input" "$root/temp" "$root/outputs" "$root/user" "$root/logs"
exec bash "$root/scripts/with_local_storage.sh" env HIP_VISIBLE_DEVICES=0 \
  "$root/.venv-moss/bin/python" "$root/runtime/ComfyUI/main.py" \
  --listen 127.0.0.1 --port "$port" \
  --base-directory "$root" \
  --models-directory "$root/models" \
  --output-directory "$root/outputs" \
  --input-directory "$root/input" \
  --temp-directory "$root/temp" \
  --user-directory "$root/user" \
  --use-pytorch-cross-attention \
  --disable-comfy-compiler \
  "$@"
