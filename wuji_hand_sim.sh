#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runtime configuration
redis_ip="localhost"
hand_side="left"  # "left" or "right"
target_fps=50
glove="manus"      # "manus" or "vdhand"
retarget_config="${SCRIPT_DIR}/wuji-retargeting/example/config/retarget_${glove}_${hand_side}.yaml"
# model inference options
use_model=true
policy_tag="wuji_left_new_1"
policy_epoch=200

# Start controller
CMD=(
  python3 "${SCRIPT_DIR}/deploy_real/server_wuji_hand_sim_redis.py"
    --hand_side "${hand_side}"
    --config "${retarget_config}"
    --redis_ip "${redis_ip}"
    --target_fps "${target_fps}"
    --no_smooth
)

if [[ "${use_model}" == "true" ]]; then
  CMD+=(--use_model --policy_tag "${policy_tag}" --policy_epoch "${policy_epoch}")
fi

"${CMD[@]}"
