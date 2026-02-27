# Quick Start Guide

## 1. Upload to Cloud Server

```bash
# On your local machine
scp TWIST2_MINIMAL_VIZ.zip user@your-server:~/

# On cloud server
ssh user@your-server
cd ~
unzip TWIST2_MINIMAL_VIZ.zip
cd TWIST2_MINIMAL_VIZ
```

## 2. Setup Environment (5 minutes)

```bash
# Create conda environment
conda create -n viz python=3.10 -y
conda activate viz

# Install dependencies
pip install -r requirements.txt

# For headless servers (no display)
export MUJOCO_GL=egl
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
```

## 3. Verify Installation

```bash
python << 'EOF'
import mujoco, numpy as np, torch, onnxruntime
print("✅ Environment ready!")
print(f"  MuJoCo: {mujoco.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  PyTorch: {torch.__version__}")
print(f"  ONNX Runtime: {onnxruntime.__version__}")
EOF
```

Expected output:
```
✅ Environment ready!
  MuJoCo: 3.x.x
  NumPy: 1.x.x
  PyTorch: 2.x.x
  ONNX Runtime: 1.x.x
```

## 4. Test with Random Actions

```bash
# Generate test data
python << 'EOF'
import numpy as np
test_actions = np.random.randn(100, 35)  # 100 frames
np.save('test_actions.npy', test_actions)
print("✅ Test data created: test_actions.npy (100 frames)")
EOF

# Visualize
python visualize_actions_simple.py \
    --actions test_actions.npy \
    --policy assets/ckpts/twist2_1017_20k.onnx \
    --xml assets/g1/g1_sim2sim.xml \
    --output test_viz.mp4

# Check output
ls -lh test_viz.mp4
# Should see: test_viz.mp4 (~a few MB)
```

## 5. Use with Your Predicted Actions

### Body Actions (35D)
```python
# In your training script
import numpy as np

# After validation
val_body_preds = your_body_policy.predict(val_obs)  # shape: (T, 35)
np.save('val_body_predictions.npy', val_body_preds)
```

Then visualize:
```bash
python visualize_actions_simple.py \
    --actions val_body_predictions.npy \
    --policy assets/ckpts/twist2_1017_20k.onnx \
    --xml assets/g1/g1_sim2sim.xml \
    --output val_body_viz.mp4 \
    --fps 30
```

### Hand Actions (20D)
```python
# After validation
val_hand_preds = your_hand_policy.predict(val_obs)  # shape: (T, 20)
np.save('val_left_hand.npy', val_hand_preds)
```

Then visualize:
```bash
python visualize_hand_actions.py \
    --actions val_left_hand.npy \
    --hand_side left \
    --output val_hand_viz.mp4 \
    --fps 30
```

## 6. Integrate into Training Loop

```python
import subprocess
import numpy as np

def visualize_epoch(predictions, epoch):
    """Visualize predictions and return video path."""
    pred_path = f'logs/pred_epoch_{epoch}.npy'
    video_path = f'logs/viz_epoch_{epoch}.mp4'

    np.save(pred_path, predictions)

    cmd = [
        'python', 'visualize_actions_simple.py',
        '--actions', pred_path,
        '--policy', 'assets/ckpts/twist2_1017_20k.onnx',
        '--xml', 'assets/g1/g1_sim2sim.xml',
        '--output', video_path,
        '--fps', '30'
    ]

    subprocess.run(cmd, check=True)
    return video_path

# In training loop
for epoch in range(num_epochs):
    # Training...
    train_loss = train_one_epoch()

    # Validation every N epochs
    if epoch % 5 == 0:
        val_preds = policy.predict(val_obs)
        video_path = visualize_epoch(val_preds, epoch)
        print(f"✅ Visualization saved: {video_path}")

        # Optional: log to wandb/tensorboard
        # import wandb
        # wandb.log({"val_viz": wandb.Video(video_path, fps=30)})
```

## Troubleshooting

### Error: `GLFW initialization failed`
```bash
export MUJOCO_GL=egl  # Use this before running
```

### Slow rendering
```bash
# Check which renderer is being used
python -c "from mujoco import Renderer; print(Renderer.get_available_renderers())"

# Should see: ['glfw', 'egl', 'osmesa']
# egl is fastest (GPU), osmesa is slowest (CPU software rendering)
```

### Out of memory
```bash
# Reduce batch size or visualize fewer frames
python visualize_actions_simple.py \
    --actions val_predictions.npy \
    --policy assets/ckpts/twist2_1017_20k.onnx \
    --xml assets/g1/g1_sim2sim.xml \
    --output val_viz.mp4 \
    --fps 30
    # Add: --width 480 --height 360  (lower resolution)
```

---

**Ready to go!** 🚀
