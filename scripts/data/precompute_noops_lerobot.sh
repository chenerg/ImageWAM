#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

config_values="$(
  imagewam_python -c 'from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["data=robotwin_omnigen2"])
resolved = OmegaConf.to_container(cfg.data, resolve=True)
print(resolved["robotwin_root"])
print(resolved["nonidle_filter_path"])
'
)"
robotwin_root="$(printf '%s\n' "${config_values}" | sed -n '1p')"
nonidle_filter_path="$(printf '%s\n' "${config_values}" | sed -n '2p')"

imagewam_print_config robotwin_root nonidle_filter_path
imagewam_run imagewam_python scripts/data/compute_robotwin_nonidle_ranges.py \
  "${robotwin_root}" \
  --output "${nonidle_filter_path}" \
  "$@"
