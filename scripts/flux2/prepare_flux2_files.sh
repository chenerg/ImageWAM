#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/checkpoints}"
FLUX2_ROOT="${FLUX2_ROOT:-${MODEL_ROOT}/flux2}"
DOWNLOAD_9B="${DOWNLOAD_9B:-true}"
HF_HUB_SSL_VERIFY="${HF_HUB_SSL_VERIFY:-true}"

FLUX2_4B_DIR="${FLUX2_4B_DIR:-${FLUX2_ROOT}/FLUX.2-klein-base-4B}"
FLUX2_9B_DIR="${FLUX2_9B_DIR:-${FLUX2_ROOT}/FLUX.2-klein-base-9B}"
FLUX2_AE_DIR="${FLUX2_AE_DIR:-${FLUX2_ROOT}/FLUX.2-dev}"

imagewam_hf_download() {
  local repo_id="$1"
  local filename="$2"
  local local_dir="$3"

  case "${HF_HUB_SSL_VERIFY}" in
    false|False|FALSE|0|no|No|NO)
      HF_HUB_DOWNLOAD_REPO_ID="${repo_id}" \
      HF_HUB_DOWNLOAD_FILENAME="${filename}" \
      HF_HUB_DOWNLOAD_LOCAL_DIR="${local_dir}" \
      imagewam_run imagewam_python - <<'PY'
import os

import requests
import urllib3
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import configure_http_backend


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
configure_http_backend(backend_factory=backend_factory)

hf_hub_download(
    repo_id=os.environ["HF_HUB_DOWNLOAD_REPO_ID"],
    filename=os.environ["HF_HUB_DOWNLOAD_FILENAME"],
    local_dir=os.environ["HF_HUB_DOWNLOAD_LOCAL_DIR"],
)
PY
      ;;
    *)
      imagewam_run huggingface-cli download "${repo_id}" "${filename}" --local-dir "${local_dir}"
      ;;
  esac
}

mkdir -p "${FLUX2_4B_DIR}" "${FLUX2_AE_DIR}"
imagewam_print_config FLUX2_4B_DIR FLUX2_AE_DIR FLUX2_9B_DIR DOWNLOAD_9B HF_HUB_SSL_VERIFY

imagewam_hf_download black-forest-labs/FLUX.2-klein-base-4B \
  flux-2-klein-base-4b.safetensors \
  "${FLUX2_4B_DIR}"

# Gated repo: requires Hugging Face access.
imagewam_hf_download black-forest-labs/FLUX.2-dev \
  ae.safetensors \
  "${FLUX2_AE_DIR}"

if [ "${DOWNLOAD_9B}" = "true" ]; then
  mkdir -p "${FLUX2_9B_DIR}"
  # Gated repo: requires Hugging Face access.
  imagewam_hf_download black-forest-labs/FLUX.2-klein-base-9B \
    flux-2-klein-base-9b.safetensors \
    "${FLUX2_9B_DIR}"
fi

cat <<EOF
Suggested environment variables:
  FLUX2_MODEL_PATH=${FLUX2_4B_DIR}/flux-2-klein-base-4b.safetensors
  FLUX2_AE_MODEL_PATH=${FLUX2_AE_DIR}/ae.safetensors
EOF
