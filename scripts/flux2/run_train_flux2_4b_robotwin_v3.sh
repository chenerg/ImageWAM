#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

# Dedicated entrypoint for FLUX.2 Klein 4B training on the RoboTwin 2.0
# LeRobot v3 dataset. Override ROBOTWIN_ROOT before invoking this script if
# your dataset lives elsewhere.
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/robotwin2.0}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${DATA_ROOT}/robotwin2.0}"
export NONIDLE_FILTER_PATH="${NONIDLE_FILTER_PATH:-${ROBOTWIN_ROOT}/nonidle_ranges.json}"

export TASK_TYPE="robotwin"
export FLUX2_VARIANT="4b"

bash "${SCRIPT_DIR}/run_train_flux2_klein_imagewam.sh" \
  data=robotwin_omnigen2_v3 \
  "data.train.dataset_dirs=[${ROBOTWIN_ROOT}]" \
  "data.val.dataset_dirs=[${ROBOTWIN_ROOT}]" \
  "data.train.nonidle_filter_path=${NONIDLE_FILTER_PATH}" \
  "data.val.nonidle_filter_path=${NONIDLE_FILTER_PATH}" \
  "$@"
