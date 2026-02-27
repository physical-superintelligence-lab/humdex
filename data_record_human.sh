#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/deploy_real"

channel="twist2"   # twist2 | sonic
sonic_body_backend="zmq"  # redis | zmq
body_zmq_ip="127.0.0.1"
body_zmq_port=5556
body_zmq_topic="pose"
data_frequency=30
task_name_base=$(date +"%Y%m%d_%H%M")
task_name="${task_name_base}_${channel}"


python server_data_record_human.py \
  --channel "${channel}" \
  --sonic_body_backend "${sonic_body_backend}" \
  --body_zmq_ip "${body_zmq_ip}" \
  --body_zmq_port "${body_zmq_port}" \
  --body_zmq_topic "${body_zmq_topic}" \
  --task_name "${task_name}" \
  --redis_ip localhost \
  --frequency "${data_frequency}" \
  --rs_w 640 \
  --rs_h 480 \
  --rs_fps 30 \
  "$@"
