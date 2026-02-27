# TWIST2 Minimal Visualization Package

Minimal files for visualizing predicted actions on cloud servers.

## Contents

```
TWIST2_MINIMAL_VIZ/
├── README.md                          # This file
├── QUICKSTART.md                      # Quick start guide
├── requirements.txt                   # Python dependencies
├── visualize_actions_simple.py        # Body visualization (35D)
├── visualize_hand_actions.py          # Hand visualization (20D)
├── assets/
│   ├── ckpts/
│   │   └── twist2_1017_20k.onnx      # Low-level RL policy (3MB)
│   └── g1/
│       ├── g1_sim2sim.xml             # G1 MuJoCo model
│       └── meshes/*.STL               # 3D meshes (~55MB)
└── wuji_retargeting/
    └── example/utils/mujoco-sim/model/
        ├── left.xml                   # Left Wuji hand model
        ├── right.xml                  # Right Wuji hand model
        └── meshes/*.STL               # Hand meshes (~5MB)
```

**Total size**: ~66MB (body + hand)

## Setup

### 1. Environment
```bash
conda create -n viz python=3.10 -y
conda activate viz
pip install -r requirements.txt
```

### 2. For headless servers (no display)
```bash
export MUJOCO_GL=egl
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
```

## Usage

### Body Visualization (35D actions)
```bash
python visualize_actions_simple.py \
    --actions predicted_body_actions.npy \
    --policy assets/ckpts/twist2_1017_20k.onnx \
    --xml assets/g1/g1_sim2sim.xml \
    --output body_viz.mp4
```

**Input format**:
- Shape: `(num_frames, 35)`
- 35D = `[vx, vy, z, roll, pitch, yaw_vel, joint_pos_0, ..., joint_pos_28]`

### Hand Visualization (20D actions)
```bash
# Left hand
python visualize_hand_actions.py \
    --actions predicted_left_hand.npy \
    --hand_side left \
    --output left_hand_viz.mp4

# Right hand
python visualize_hand_actions.py \
    --actions predicted_right_hand.npy \
    --hand_side right \
    --output right_hand_viz.mp4
```

**Input format**:
- Shape: `(num_frames, 20)` or `(num_frames, 5, 4)`
- 20D = 5 fingers × 4 joints

### Example: Preparing Actions
```python
import numpy as np

# Body actions
body_actions = your_policy.predict(obs)  # shape: (T, 35)
np.save('predicted_body_actions.npy', body_actions)

# Hand actions
hand_actions = your_hand_policy.predict(obs)  # shape: (T, 20)
np.save('predicted_left_hand.npy', hand_actions)
```

### Complete Example
```bash
# 1. Prepare your predicted actions
python << 'EOF'
import numpy as np
# Example: random actions for testing
actions = np.random.randn(100, 35)  # 100 frames
np.save('test_actions.npy', actions)
EOF

# 2. Visualize
python visualize_actions_simple.py \
    --actions test_actions.npy \
    --policy assets/ckpts/twist2_1017_20k.onnx \
    --xml assets/g1/g1_sim2sim.xml \
    --output test_viz.mp4 \
    --fps 30

# 3. Check output
ls -lh test_viz.mp4
```

## Verification

```bash
# Test environment
python << 'EOF'
import mujoco, numpy as np, torch, onnxruntime
print("✅ All packages OK")
print(f"MuJoCo: {mujoco.__version__}")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
EOF
```

## Troubleshooting

### Error: `GLFW initialization failed`
```bash
export MUJOCO_GL=egl  # Use EGL renderer (no display needed)
```

### Error: `No module named 'imageio'`
```bash
pip install imageio imageio-ffmpeg
```

### Slow rendering
```bash
# Use EGL instead of software rendering
export MUJOCO_GL=egl  # Requires GPU
# If no GPU available, OSMesa will be used (slower)
```

## Notes

- **Policy**: The ONNX file contains the complete low-level RL policy (architecture + weights + normalizer)
- **No training code needed**: ONNX is self-contained
- **Lightweight**: Only ~61MB total (vs ~7GB+ for full training environment)
- **Cross-platform**: Works on any platform with MuJoCo 3.0+

## Training Integration

Use in your training loop:
```python
import subprocess
import numpy as np

def visualize_validation(val_predictions, epoch, output_dir='logs'):
    pred_path = f'{output_dir}/val_pred_epoch_{epoch}.npy'
    video_path = f'{output_dir}/val_viz_epoch_{epoch}.mp4'

    np.save(pred_path, val_predictions)

    subprocess.run([
        'python', 'visualize_actions_simple.py',
        '--actions', pred_path,
        '--policy', 'assets/ckpts/twist2_1017_20k.onnx',
        '--xml', 'assets/g1/g1_sim2sim.xml',
        '--output', video_path
    ])

    return video_path

# In training loop
for epoch in range(num_epochs):
    # ... training ...

    if epoch % 5 == 0:
        val_preds = policy.predict(val_obs)
        viz_path = visualize_validation(val_preds, epoch)
        # Log to wandb/tensorboard
        # wandb.log({"val_viz": wandb.Video(viz_path)})
```

---

**Package Version**: v1.0
**Date**: 2025-12-29
**TWIST2 Team**
