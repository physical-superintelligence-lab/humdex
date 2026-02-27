# TWIST2_github (Draft README v1)

This README is the first cleaned-up version for the current open-source layout.

Current scope:
- Installation
- Unified teleop entry
- sim2sim / sim2real quick start
- wuji hand quick start

---

## Content Table

- [Installation](#installation)
- [Teleop](#teleop)
- [Sim2Sim and Sim2Real](#sim2sim-and-sim2real)
- [Wuji Hand](#wuji-hand)
- [Camera and Data Collection](#camera-and-data-collection)
- [Troubleshooting](#troubleshooting)

---

## Installation

We currently use two environments:

- `gmr`: teleop runtime (`teleop.sh`)
- `humdex`: non-teleop modules (sim/deploy/training side, to be documented in detail later)

### 1) Prerequisites

```bash
sudo apt update
sudo apt install -y git build-essential cmake python3-dev redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

### 2) Repository Layout (for Sonic)

For `--policy sonic`, place repos as siblings:

```text
<workspace_root>/
  TWIST2_github/
  GR00T-WholeBodyControl/
```

Reason: sonic ZMQ message packing imports `gear_sonic` from `GR00T-WholeBodyControl`.

### 3) Create `gmr` Environment (teleop)

```bash
conda create -n gmr python=3.10 -y
conda activate gmr

cd /path/to/TWIST2_github/GMR
pip install -e .

cd /path/to/TWIST2_github
pip install numpy scipy redis pyzmq python-osc
conda install -c conda-forge libstdcxx-ng -y
```

### 4) Create `humdex` Environment (non-teleop)

```bash
conda create -n humdex python=3.8 -y
conda activate humdex

cd /path/to/TWIST2_github
pip install -e ./rsl_rl
pip install -e ./legged_gym
pip install -e ./pose
```

### 5) Quick Checks

```bash
conda activate gmr
python -c "import numpy, scipy, redis, zmq; print('gmr ok')"

conda activate humdex
python -c "import numpy, redis; print('humdex ok')"
```

---

## Teleop

### 1) Unified Entry

Use root entry:

```bash
bash teleop.sh [options] [-- extra_args]
```

Supported selectors:

- `--policy {twist2|sonic}` (default `twist2`)
- `--body_source {vdmocap|slimevr}` (default `vdmocap`)
- `--hand_source {vdhand|manus}` (default `vdhand`)

Aliases:

- `--body` for `--body_source`
- `--hand` for `--hand_source`

Execution mode:

- `--dry-run` (default): print resolved pipeline only
- `--run`: execute pipeline

### 2) Common Examples

Dry run (default combo):

```bash
bash teleop.sh
```

Run default combo:

```bash
bash teleop.sh --run
```

Run sonic + vdmocap + manus:

```bash
bash teleop.sh --policy sonic --body_source vdmocap --hand_source manus --run
```

Run twist2 + slimevr + vdhand:

```bash
bash teleop.sh --policy twist2 --body_source slimevr --hand_source vdhand --run
```

Forward extra arguments to runtime parser:

```bash
bash teleop.sh --policy sonic --body vdmocap --hand manus --run -- --print_every 300
```

### 3) Config Files

Main YAMLs:

- `deploy_real/config/teleop_twist2_vdmocap_vdhand.yaml`
- `deploy_real/config/teleop_sonic_vdmocap_vdhand.yaml`

These YAMLs contain runtime parameters (target fps, ramp durations, hand mode, source ports, etc.).

Current notable fields:

- Ramp: `start_ramp_seconds`, `toggle_ramp_seconds`, `exit_ramp_seconds`, `ramp_ease`
- Manus: `manus_address`, `manus_left_sn`, `manus_right_sn`, `manus_auto_assign`
- SlimeVR/VMC: `vmc_ip`, `vmc_port`, `vmc_use_fk`, `vmc_bvh_path`
- Loop: `print_every`, `max_steps`

### 4) Keyboard Behavior (Current)

- `k`: toggle send/default mode
- `p`: toggle hold mode

For sonic, body ZMQ output is aligned with ramped body state (same k/p transition semantics as published body state).

---

## Sim2Sim and Sim2Real

This section covers low-level controller bring-up scripts currently in repo root.

### 1) Sim2Sim

Script: `sim2sim.sh`

```bash
bash sim2sim.sh
```

What it does:
- launches `deploy_real/server_low_level_g1_sim.py`
- uses ONNX policy at `assets/ckpts/twist2_1017_20k.onnx`
- uses sim XML `assets/g1/g1_sim2sim_29dof.xml`
- default device is `cpu`

### 2) Sim2Real

Script: `sim2real.sh`

```bash
bash sim2real.sh
```

What it does:
- launches `deploy_real/server_low_level_g1_real.py`
- uses ONNX policy at `assets/ckpts/twist2_1017_20k.onnx`
- uses configured network interface (`net=...`) for robot connection
- default device is `cpu`

Important:
- edit `net` in `sim2real.sh` to your real NIC name before running
- script currently activates `twist2` env; if you standardize on `humdex`, update that line

### 3) Offline Motion Stream (Optional)

Script: `run_motion_server.sh`

```bash
bash run_motion_server.sh
```

What it does:
- runs `deploy_real/server_motion_lib.py`
- reads motion from `assets/example_motions/*.pkl`
- publishes high-level motion to Redis (`redis_ip` in script, default `localhost`)

---

## Wuji Hand

### 1) Real Hand Controller via Redis

Script: `wuji_hand_redis_single.sh`

```bash
bash wuji_hand_redis_single.sh
```

What it does:
- runs `deploy_real/server_wuji_hand_redis.py`
- reads `hand_tracking_*` + `wuji_hand_mode_*` from Redis
- retargets with config under `wuji-retargeting/example/config/`
- writes commands to real Wuji hand hardware

Current defaults in script:
- `hand_side=left`
- `redis_ip=localhost`
- `target_fps=50`

### 2) Sim Visualization via Redis

Script: `wuji_hand_redis_single_sim.sh`

```bash
bash wuji_hand_redis_single_sim.sh
```

What it does:
- runs `deploy_real/server_wuji_hand_sim_redis.py`
- reads same Redis hand keys
- visualizes retarget result in MuJoCo (no real hardware actuation)

Current defaults in script:
- `hand_side=right`
- `redis_ip=localhost`
- `target_fps=60`

---

## Camera and Data Collection

This section documents the current scripts:

- `realsense_zmq_pub_g1.sh`
- `data_record.sh` (keyboard-controlled demo recording)
- `data_record_human.sh` (human recording entry)

### 1) Start Camera Stream on g1

Script: `realsense_zmq_pub_g1.sh`

```bash
bash realsense_zmq_pub_g1.sh
```

What it does:
- SSH to `--host`
- activate `--conda_env`
- `cd --remote_dir/deploy_real`
- `killall videohub_pc4` (best effort)
- run `server_realsense_zmq_pub.py` with passthrough args

Notes:
- default publish endpoint is `0.0.0.0:5555`
- all extra args after script options are forwarded to `server_realsense_zmq_pub.py`
- override only when needed, e.g. `bash realsense_zmq_pub_g1.sh --host g1 --remote_dir ~/TWIST2_github`

### 2) Keyboard Data Recording

Script: `data_record.sh`

```bash
bash data_record.sh

# sonic channel
bash data_record.sh --channel sonic
```

Behavior:
- default `task_name` is `YYYYMMDD_HHMM_<channel>`
- records vision (ZMQ), Redis state/action data, and episode video
- keyboard backend defaults are configured in script (`evdev` + footswitch device)

### 3) Human Data Recording

Script: `data_record_human.sh`

```bash
bash data_record_human.sh

# sonic channel
bash data_record_human.sh --channel sonic
```

Behavior:
- default `task_name` is `YYYYMMDD_HHMM_<channel>`
- supports manual override: `--task_name your_name`
- records with `server_data_record_human.py`

### 4) Sonic Recording Path (Current)

When `--channel sonic --sonic_body_backend zmq`:

- body data is read from ZMQ (`pose` topic, configurable IP/port/topic)
- hand-related data remains from Redis (`hand_tracking_*`, wuji hand keys, etc.)

---

## Troubleshooting

- `ModuleNotFoundError: general_motion_retargeting`
  - activate `gmr`
  - run `pip install -e /path/to/TWIST2_github/GMR`

- `python-osc unavailable` when using `--body_source slimevr`
  - install `python-osc` in `gmr`

- `pack_pose_message_not_found` for sonic ZMQ
  - ensure sibling repo exists: `../GR00T-WholeBodyControl`
  - ensure `gear_sonic` import works in current env

- `hand_frame_status=no_update` with `manus`
  - verify `manus_address` and glove SN settings in YAML

- `body_frame_status=no_update` with `vdmocap`
  - verify mocap sender IP/port/index and network route

- `sdk_error=[Errno 98] Address already in use` with `--body_source slimevr`
  - another process is already binding the same VMC port (default `39539`)

- `pack_pose_message_not_found` while old sonic script works
  - check that `GR00T-WholeBodyControl` is sibling to this repo
  - old scripts also depend on `gear_sonic`; they just inject that path explicitly

---
