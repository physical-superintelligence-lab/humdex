SCRIPT_DIR=$(dirname $(realpath $0))
cd deploy_real

# Runtime configuration
redis_ip="localhost"
hand_side="left"  # "left" or "right"
target_fps=50
serial_number="3555374E3533"
policy_tag="wuji_left_new_1"
policy_epoch=200

# Start controller
python ../deploy_real/server_wuji_hand_redis.py \
    --hand_side ${hand_side} \
    --serial_number ${serial_number} \
    --redis_ip ${redis_ip} \
    --target_fps ${target_fps} \
    --no_smooth \
    --use_model \
    --policy_tag ${policy_tag} \
    --policy_epoch ${policy_epoch}
