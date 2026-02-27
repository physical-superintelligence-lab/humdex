# Installation (Draft v1)

This document follows the current cleanup plan:

- `teleop` uses the `gmr` environment.
- Other modules (sim2sim/sim2real/wuji/data recording) use the `humdex` environment.
- `sonic` depends on a sibling repo: `GR00T-WholeBodyControl`.

---

## 1) Prerequisites

- Ubuntu 22.04 (recommended)
- Miniconda/Anaconda available (`conda` command works)
- Git
- Redis server
- Python build basics:

```bash
sudo apt update
sudo apt install -y git build-essential cmake python3-dev
```

Install and start Redis (if not already installed):

```bash
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

---

## 2) Repository Placement (Required for Sonic)

For sonic teleop scripts in this repo, the GR00T repo should be placed as a sibling directory:

```text
<workspace_root>/
  TWIST2_github/
  GR00T-WholeBodyControl/
```

Example:

```text
/home/hongyijing/heng/
  TWIST2_github/
  GR00T-WholeBodyControl/
```

Why this matters:

- several sonic scripts resolve GR00T path as `../GR00T-WholeBodyControl`.

---

## 3) Create `humdex` Environment (non-teleop modules)

`humdex` is the environment name used by the new architecture docs.
Current scripts may still reference `twist2`; for now we keep both names compatible.

Create the env:

```bash
conda create -n humdex python=3.8 -y
conda activate humdex
```

Install core packages (baseline from current README/workflows):

```bash
cd /home/hongyijing/heng/TWIST2_github

pip install -e ./rsl_rl
pip install -e ./legged_gym
pip install -e ./pose

pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor pyzmq
pip install redis[hiredis]
pip install pyttsx3
pip install onnx onnxruntime-gpu
pip install customtkinter
```

Notes:

- If your workflow needs Isaac Gym, install it separately in this env.
- Existing scripts that still say `activate twist2` can be migrated later to `humdex`.

---

## 4) Create `gmr` Environment (teleop modules)

Create env:

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
```

Install GMR from local repo clone (recommended in this workspace):

```bash
cd /home/hongyijing/heng/TWIST2_github/GMR
pip install -e .
```

Install teleop runtime dependencies:

```bash
cd /home/hongyijing/heng/TWIST2_github
pip install numpy scipy redis pyzmq
conda install -c conda-forge libstdcxx-ng -y
```

For SlimeVR/VMC input:

```bash
pip install python-osc
```

For PICO/XRobot teleop path (if needed):

- install XRoboToolkit PC service and SDK binding as in `TWIST2_README.md`.

---

## 5) Sonic Dependency Setup (`GR00T-WholeBodyControl`)

Sonic teleop in `TWIST2_github` needs GR00T utilities (e.g., pose message packing).

Basic setup:

```bash
cd /home/hongyijing/heng
git clone <your-gr00t-repo-url> GR00T-WholeBodyControl
```

Then install GR00T dependencies following GR00T docs (inside that repo).  
At minimum, ensure `gear_sonic` Python modules are importable in teleop runtime.

Quick import check (in `gmr` env):

```bash
conda activate gmr
python -c "import sys, pathlib; p=pathlib.Path('/home/hongyijing/heng/GR00T-WholeBodyControl'); print(p.exists())"
```

---

## 6) Quick Environment Checks

Check teleop env:

```bash
conda activate gmr
python -c "import redis, zmq, numpy, scipy; print('gmr env ok')"
```

Check non-teleop env:

```bash
conda activate humdex
python -c "import redis, numpy; print('humdex env ok')"
```

Check repo-local commands:

```bash
cd /home/hongyijing/heng/TWIST2_github
bash sim2real.sh --help || true
```

---

## 7) Naming Migration Note (`twist2` -> `humdex`)

Current state in repo:

- many scripts still call `source ~/miniconda3/bin/activate twist2`.

Target state:

- non-teleop scripts should eventually use `humdex`.

Recommended transition:

1. keep current scripts runnable
2. introduce new entrypoints using `humdex`
3. migrate old scripts in a controlled PR

---

## 8) Troubleshooting

- `ModuleNotFoundError: general_motion_retargeting`
  - ensure `gmr` env active
  - run `pip install -e /path/to/TWIST2_github/GMR`
- `ImportError` on sonic ZMQ packers
  - verify sibling repo path `../GR00T-WholeBodyControl`
  - ensure GR00T dependencies installed
- `redis connection failed`
  - check `systemctl status redis-server`
  - verify `--redis_ip` and network route

