#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
TASK_TYPE="${TASK_TYPE:-robotwin}"
TASK="${TASK:-robotwin_omnigen2_imagewam}"
ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_omnigen2_init.pt}"

imagewam_require_env OMNIGEN2_SRC
imagewam_require_env OMNIGEN2_MODEL_PATH
imagewam_require_env QWEN_MODEL_PATH
imagewam_require_env RESUME_PATH

imagewam_print_config TASK RESUME_PATH
TASK="${TASK}" imagewam_run bash scripts/omnigen2/train_omnigen2_imagewam.sh "${GPU_PER_NODE}" \
  model.omnigen2_model_path="${OMNIGEN2_MODEL_PATH}" \
  model.omnigen2_vae_path="${OMNIGEN2_MODEL_PATH}" \
  model.qwen_path="${QWEN_MODEL_PATH}" \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  resume="${RESUME_PATH}" \
  "$@"
