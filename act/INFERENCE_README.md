# Policy Inference Guide

## Quick Start

All commands run from `deploy_real/` directory:

```bash
cd deploy_real
```

### 1. Replay GT Actions (Sim Only)
Visualize ground truth actions from dataset in simulation. No robot needed.

```bash
python policy_inference.py replay \
    --dataset ../act/data/dataset.hdf5 \
    --episode 0 \
    --sim_only \
    --output replay_ep0.mp4
```

### 2. Replay GT Actions (To Robot)
Publish ground truth actions to Redis for robot execution.

```bash
python policy_inference.py replay \
    --dataset ../act/data/dataset.hdf5 \
    --episode 0 \
    --redis_ip localhost
```

### 3. Evaluate Policy Offline
Run policy with dataset observations, compare predicted vs GT in sim.

```bash
python policy_inference.py eval \
    --ckpt_dir ../act/ckpts/my_run \
    --dataset ../act/data/dataset.hdf5 \
    --episode 0 \
    --temporal_agg \
    --chunk_size 50 \
    --hidden_dim 512 \
    --output eval_ep0.mp4
```

**Outputs:**
- `eval_ep0.mp4` - Side-by-side video (predicted | GT)
- `eval_ep0_actions.npy` - Predicted actions
- `eval_ep0_gt_actions.npy` - GT actions
- MSE/MAE metrics printed

### 4. Real-time Inference
Run policy with live observations from robot.

```bash
python policy_inference.py infer \
    --ckpt_dir ../act/ckpts/my_run \
    --temporal_agg \
    --chunk_size 50 \
    --hidden_dim 512 \
    --redis_ip localhost \
    --vision_ip 192.168.123.164
```

---

## Mode Summary

| Mode | Redis | Robot | Sim Viz | Description |
|------|-------|-------|---------|-------------|
| `replay --sim_only` | - | - | Yes | Visualize GT actions in sim |
| `replay` | Yes | Yes | - | Publish GT actions to robot |
| `eval` | - | - | Yes | Test policy with dataset obs |
| `infer` | Yes | Yes | - | Real-time policy control |

---

## Policy Config Parameters

These must match your training config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk_size` | 50 | Action chunk size (num_queries) |
| `--hidden_dim` | 512 | Transformer hidden dimension |
| `--temporal_agg` | False | Enable temporal aggregation |

Note: `dim_feedforward` is auto-set to `hidden_dim * 4`.

---

## Code Structure

```
deploy_real/policy_inference.py
├── Config              # Dataclass for all configuration
├── RedisIO             # Read state / publish actions via Redis
├── VisionReader        # Read images via ZMQ from robot camera
├── ACTPolicyWrapper    # Load policy, preprocess, inference, postprocess
├── DatasetReader       # Load episodes from HDF5
├── replay_episode()    # Replay mode implementation
├── eval_offline()      # Eval mode implementation
└── run_inference()     # Infer mode implementation
```

---

## Key Logic

### Normalization (must match training)
```python
# Preprocess (before policy)
qpos_norm = (qpos - stats['qpos_mean']) / stats['qpos_std']
image_norm = image / 255.0  # (1, 1, C, H, W)

# Postprocess (after policy)
action = action_norm * stats['action_std'] + stats['action_mean']
```

### Temporal Aggregation
When `--temporal_agg` is enabled:
- Policy is queried every timestep
- Multiple predictions for current timestep are weighted by exponential decay
- Newer predictions have higher weight (k=0.01)

Without temporal aggregation:
- Policy queried every `chunk_size` steps
- Use cached action sequence

### Data Dimensions
- **qpos (state)**: 54D = state_body(34) + state_wuji_hand_left(20)
- **action**: 55D = action_body(35) + action_wuji_qpos_target_left(20)
- **image**: (720, 1280, 3) RGB

### Redis Keys
```
# State (read)
state_body_unitree_g1_with_hands          # 34D
state_wuji_hand_left_unitree_g1_with_hands # 20D

# Action (publish)
action_body_unitree_g1_with_hands         # 35D
action_neck_unitree_g1_with_hands         # 2D
t_action                                   # timestamp (ms)
```

---

## Checkpoint Directory Structure

Your `ckpt_dir` should contain:
```
ckpts/my_run/
├── policy_best.ckpt      # or policy_last.ckpt
└── dataset_stats.pkl     # normalization stats from training
```

---

## Troubleshooting

**Policy loading fails with shape mismatch:**
- Check `--hidden_dim` and `--chunk_size` match training

**Sim visualization fails:**
- Ensure `act/sim_viz/` exists with `visualizers.py`
- Check ONNX policy path in `sim_viz/assets/`

**No state data from Redis:**
- Verify robot low-level server is running
- Check Redis IP and port
