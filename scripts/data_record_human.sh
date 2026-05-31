#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../deploy_real"

channel="twist2"   # twist2 | sonic
redis_ip="localhost"

data_frequency=30
task_name_base=$(date +"%Y%m%d_%H%M")
task_name="${task_name_base}_${channel}"

glove="manus"  # manus | vdhand
retarget_config_left="${SCRIPT_DIR}/../wuji-retargeting/example/config/retarget_${glove}_left.yaml"
retarget_config_right="${SCRIPT_DIR}/../wuji-retargeting/example/config/retarget_${glove}_right.yaml"


python server_data_record_human.py \
  --channel "${channel}" \
  --task_name "${task_name}" \
  --redis_ip "${redis_ip}" \
  --frequency "${data_frequency}" \
  --rs_w 640 --rs_h 480 --rs_fps 30 \
  --local_wuji_retarget_config_left "${retarget_config_left}" \
  --local_wuji_retarget_config_right "${retarget_config_right}" \
  "$@"
