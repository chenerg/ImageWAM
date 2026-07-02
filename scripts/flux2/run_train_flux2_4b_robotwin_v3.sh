#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

# Dedicated entrypoint for FLUX.2 Klein 4B training on the RoboTwin 2.0
# LeRobot v3 dataset. Edit configs/data/robotwin_omnigen2.yaml if your
# dataset lives elsewhere.

export TASK_TYPE="robotwin"
export FLUX2_VARIANT="4b"

bash "${SCRIPT_DIR}/run_train_flux2_klein_imagewam.sh" \
  data=robotwin_omnigen2_v3 \
  "$@"
