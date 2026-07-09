#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
TASK_TYPE="${TASK_TYPE:-robotwin}"        # libero | robotwin
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}"      # 4b | 9b
ZERO_STAGE="${ZERO_STAGE:-1}"             # 1 | zero1 | 2 | zero2
PRECOMPUTE_QWEN3_CACHE="${PRECOMPUTE_QWEN3_CACHE:-false}"
USE_CLEAN_ROBOTWIN="${USE_CLEAN_ROBOTWIN:-false}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"

imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_AE_MODEL_PATH

case "${FLUX2_VARIANT}" in
  4b)
    MODEL_CONFIG="configs/model/imagewam_flux2_klein_4b_base.yaml"
    TASK_SUFFIX="flux2_klein_4b_base_imagewam"
    FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
    FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
    ;;
  9b)
    MODEL_CONFIG="configs/model/imagewam_flux2_klein_9b_base.yaml"
    TASK_SUFFIX="flux2_klein_9b_base_imagewam"
    FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-8B}"
    FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${MODEL_ROOT}/flux2/FLUX.2-klein-base-9B/flux-2-klein-base-9b.safetensors}"
    ;;
  *) echo "Invalid FLUX2_VARIANT=${FLUX2_VARIANT}; expected 4b or 9b" >&2; exit 1 ;;
esac
export FLUX2_MODEL_PATH FLUX2_QWEN3_MODEL_SPEC ZERO_STAGE

case "${TASK_TYPE}" in
  libero)
    ACTION_DIM=7
    TASK_NAME="libero_${TASK_SUFFIX}"
    ;;
  robotwin)
    ACTION_DIM=14
    TASK_NAME="robotwin_${TASK_SUFFIX}"
    if [ "${USE_CLEAN_ROBOTWIN}" = "true" ]; then
      TASK_NAME="${TASK_NAME/_imagewam/_clean_imagewam}"
    elif [ "${USE_CLEAN_ROBOTWIN}" != "false" ]; then
      echo "Invalid USE_CLEAN_ROBOTWIN=${USE_CLEAN_ROBOTWIN}; expected true or false" >&2
      exit 1
    fi
    ;;
  *) echo "Invalid TASK_TYPE=${TASK_TYPE}; expected libero or robotwin" >&2; exit 1 ;;
esac

ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_flux2_${FLUX2_VARIANT}_${TASK_TYPE}_init.pt}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

imagewam_print_config TASK_TYPE TASK_NAME FLUX2_VARIANT FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH ACTION_INIT

if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python scripts/flux2/preprocess_action_dit_flux2.py \
    --model-config "${MODEL_CONFIG}" \
    --flux2-src-path "${FLUX2_SRC}" \
    --flux2-model-path "${FLUX2_MODEL_PATH}" \
    --variant "klein-base-${FLUX2_VARIANT}" \
    --action-dim "${ACTION_DIM}" \
    --output "${ACTION_INIT}" \
    --apply-alpha-scaling true
fi

if [ "${PRECOMPUTE_QWEN3_CACHE}" = "true" ]; then
  imagewam_run torchrun --standalone --nproc_per_node="${GPU_PER_NODE}" \
    scripts/flux2/precompute_flux2_qwen3_embeds.py \
    task="${TASK_NAME}" \
    qwen_cache_batch_size="${QWEN_CACHE_BATCH_SIZE:-16}" \
    qwen_cache_save_workers="${QWEN_CACHE_SAVE_WORKERS:-4}" \
    qwen_cache_overwrite="${QWEN_CACHE_OVERWRITE:-false}" \
    model.variant="klein-base-${FLUX2_VARIANT}" \
    model.qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}" \
    flux2_qwen3_model_spec="${FLUX2_QWEN3_MODEL_SPEC}"
fi

COMMON_OVERRIDES=(
  "model.flux2_model_path=${FLUX2_MODEL_PATH}"
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}"
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}"
  "model.action_dit_pretrained_path=${ACTION_INIT}"
)

TASK="${TASK_NAME}" imagewam_run bash scripts/flux2/train_flux2_klein_imagewam.sh "${GPU_PER_NODE}" \
  "${COMMON_OVERRIDES[@]}" \
  "$@"
