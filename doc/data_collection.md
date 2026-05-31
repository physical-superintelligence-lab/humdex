# Data Collection Guide

Recording captures a running teleop session into episodes. Bring up teleop first
(see [`teleop.md`](teleop.md)), then start the camera and the recorder below.

## 1. Robot Data

Run the full teleop stack — G1 controller + hand controller + teleop — per [`teleop.md`](teleop.md).

### 1.1 Camera (on G1)

Plug the G1 camera USB into the G1 host, then SSH to `unitree@192.168.123.164` (password: `123`).

Copy `deploy_real/server_realsense_zmq_pub.py` to `~` on G1, then create a dedicated `realsense` conda environment and install dependencies manually.

```bash
# on local workstation (repo root)
scp deploy_real/server_realsense_zmq_pub.py unitree@192.168.123.164:~/
```

```bash
# on g1 after ssh login
conda create -y -n realsense python=3.10
conda activate realsense
python -m pip install --upgrade pip
python -m pip install pyrealsense2 pyzmq numpy opencv-python rich zmq
```

Then start the publisher from the workstation:

```bash
bash scripts/realsense_zmq_pub_g1.sh
```

### 1.2 Record

```bash
bash scripts/data_record.sh

# sonic channel
bash scripts/data_record.sh --channel sonic
```

## 2. Human Data

Recording the operator does **not** drive the robot, so no G1 / hand controller is needed —
just teleop ([`teleop.md`](teleop.md) §4) and a worn camera. 

For sonic, also start a background
sim ([`teleop.md`](teleop.md) §2, sim controller) so the encoder still produces tokens.

### 2.1 Wear the camera

Plug the RealSense USB into the workstation, then wear it using:

- a 3D-printed adapter from [link](https://drive.google.com/file/d/1PKtpvaxZI7zmqRgvxg64AasXwEP1iWsf/view?usp=sharing)
- a GoPro neck mount from [link](https://www.amazon.com/TELESIN-Magnetic-Release-Necklace-Insta360/dp/B0CZ41S1ZQ/ref=sr_1_1_sspa?crid=2G13WWF52ZKYD&dib=eyJ2IjoiMSJ9.0eDdz0m5oeRrIjXp4IIcJF_SgH-CECEHq6yza_EyAefQ9SxyBDzOkLRAI9O8PH8WH5TlGKEsGbYKtxwlwTUpwzHRxuY59GhFi_XovB6RovVXXvgDO4_AHKdq6cPyAtTxEquAhsHdMem50uXqA-LL8oU8rcHwVFneuSipbTCSxXb4Nd-6ZqDhFlMsF8q-iyrlmQSBI5TrmAGmD4Jh8GqQwzVkWDOkfZMSPRPoOblS9tg.bI1ysBZpCOwbrNvHFJ5RY34iDENihSk5FHbd0SyCth8&dib_tag=se&keywords=GoPro+Neck+Mount&qid=1772824475&sprefix=gopro+neck+mount%2Caps%2C171&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1)


### 2.2 Record

```bash
bash scripts/data_record_human.sh

# sonic channel
bash scripts/data_record_human.sh --channel sonic
```

## 3. Recording Controls

Both recorders share:

- `r`: start/stop one episode
- `q`: quit recorder

## 4. Data Layout

Both recorders save under `deploy_real/humdex_demonstration/<task_name>/`, where `<task_name>`
is `YYYYMMDD_HHMM_<channel>`. Each episode:

- `episode_0001/`
  - `rgb/` (JPEG frames, e.g. `000000.jpg`)
  - `data.json` (per-frame metadata + states/actions)

Per-frame fields in `data.json`:

- common: `idx`, `rgb`, `t_img`, `t_record_ms`; hand (`hand_tracking_*`, `state_wuji_hand_*`, `action_wuji_qpos_target_*`); timestamps (`t_*`)
- twist2: `state_body` (31), `action_body` (35)
- sonic: `state_body` (29), `action_token` (64)

## 5. Sonic Channel (Token-Level)

Versus the default twist2 channel, `--channel sonic` changes two things.

**1. An extra controller.** The body action becomes a 64-D encoder token, produced by the sonic
controller (not the teleop pipeline). So a controller must be running and publishing on ZMQ
`g1_debug`:5557 (on by default) — start it per [`teleop.md`](teleop.md) §2: `real` for robot,
`sim` for human.

**2. The body fields.** The token replaces the joint action; hands and image are unchanged:

| field          | twist2 | sonic |
| -------------- | :----: | :---: |
| `state_body`   | 31     | 29    |
| `action_body`  | 35     | —     |
| `action_token` | —      | 64    |

For human data, because `state_body` is now real, skip the `convert_human_data.py` state approximation.
