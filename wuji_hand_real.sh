#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Runtime configuration
redis_ip="localhost"
hand_side="left"  # "left" or "right"
target_fps=50
serial_number="3555374E3533"
# model inference options
use_model=true
policy_tag="wuji_left_new_1"
policy_epoch=200

# Start controller
CMD=(
  python3 "${SCRIPT_DIR}/wuji_policy/server_wuji_hand_redis.py"
    --hand_side "${hand_side}"
    --serial_number "${serial_number}"
    --redis_ip "${redis_ip}"
    --target_fps "${target_fps}"
    --no_smooth
)

if [[ "${use_model}" == "true" ]]; then
  CMD+=(--use_model --policy_tag "${policy_tag}" --policy_epoch "${policy_epoch}")
fi

"${CMD[@]}"
