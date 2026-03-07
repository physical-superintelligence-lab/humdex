#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

# Default input path: data_record.sh output root
# server_data_record.py default -> deploy_real/humdex_demonstration/<task_name>
input_path="${SCRIPT_DIR}/deploy_real/humdex_demonstration"

python wuji_data_collect.py \
  --input_path "${input_path}" \
  "$@"

