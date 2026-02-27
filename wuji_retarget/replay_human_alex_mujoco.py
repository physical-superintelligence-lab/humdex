#!/usr/bin/env python3
"""
Replay with comparison:
- GT action (from `wuji_right.npz`: action_wuji_qpos_target)
- Model inference action (GeoRT forward)

Then drive MuJoCo WujiHand sim with either GT or inference, and print error metrics.

Data (choose one):
- human_alex.npy: (T, 21, 3) MediaPipe 21 keypoints (demo)
- wuji_right.npz: (T, 5, 3) fingertips_rel_wrist + (T, 20) action_wuji_qpos_target (GT)

Sim model (default):
  wuji_retargeting/example/utils/mujoco-sim/model/{left,right}.xml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


FINGERTIP_5_IDX = np.array([4, 8, 12, 16, 20], dtype=int)  # from MediaPipe 21
WRIST_IDX = 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_mjcf(hand_side: str) -> str:
    side = hand_side.strip().lower()
    return str(
        (_repo_root() / "wuji_retargeting" / "example" / "utils" / "mujoco-sim" / "model" / f"{side}.xml").resolve()
    )


def _load_pts21_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (21, 3):
        raise ValueError(f"Expected (T, 21, 3), got {arr.shape} from {path}")
    return arr


def _load_wuji_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tips5: (T, 5, 3) float32  (fingertips_rel_wrist)
      action20: (T, 20) float32 (action_wuji_qpos_target)
    """
    npz = np.load(path, allow_pickle=True)
    if "fingertips_rel_wrist" not in npz.files or "action_wuji_qpos_target" not in npz.files:
        raise KeyError(f"{path} must contain fingertips_rel_wrist + action_wuji_qpos_target, got keys={npz.files}")
    tips5 = np.asarray(npz["fingertips_rel_wrist"], dtype=np.float32)
    action20 = np.asarray(npz["action_wuji_qpos_target"], dtype=np.float32)
    if tips5.ndim != 3 or tips5.shape[1:] != (5, 3):
        raise ValueError(f"Expected fingertips_rel_wrist (T,5,3), got {tips5.shape}")
    if action20.ndim != 2 or action20.shape[1] != 20:
        raise ValueError(f"Expected action_wuji_qpos_target (T,20), got {action20.shape}")
    if tips5.shape[0] != action20.shape[0]:
        raise ValueError(f"Time mismatch: tips5 T={tips5.shape[0]} vs action20 T={action20.shape[0]}")
    return tips5, action20


def _slice_range(n: int, start: int, end: int) -> Tuple[int, int]:
    s = int(start)
    e = int(end)
    if s < 0:
        s = 0
    if e < 0 or e > n:
        e = n
    if e <= s:
        raise ValueError(f"Invalid range: start={start}, end={end}, n={n}")
    return s, e


def _points5_from_pts21(pts21: np.ndarray, relative_to_wrist: bool) -> np.ndarray:
    pts21 = np.asarray(pts21, dtype=np.float32).reshape(21, 3)
    pts5 = pts21[FINGERTIP_5_IDX, :]
    if relative_to_wrist:
        pts5 = pts5 - pts21[WRIST_IDX : WRIST_IDX + 1, :]
    return pts5


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay human_alex.npy -> 5 fingertips -> GeoRT -> MuJoCo Wuji sim")
    ap.add_argument("--data_path", type=str, default=str(_repo_root() / "wuji_retarget" / "data" / "wuji_right.npz"))
    ap.add_argument("--hand_side", type=str, default="right", choices=["left", "right"])
    ap.add_argument("--ckpt_tag", type=str, required=True, help="GeoRT checkpoint tag, e.g. geort_filter_wuji_3")
    ap.add_argument("--epoch", type=int, default=-1, help="checkpoint epoch; -1 means latest")
    ap.add_argument("--mjcf_path", type=str, default="", help="override MuJoCo xml path")
    ap.add_argument("--fps", type=float, default=60.0, help="playback/control FPS")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--start", type=int, default=0, help="start frame (inclusive)")
    ap.add_argument("--end", type=int, default=-1, help="end frame (exclusive); -1 means end")
    ap.add_argument("--no_loop", action="store_true", help="do not loop replay")
    ap.add_argument("--relative_to_wrist", action="store_true", help="(npy only) subtract wrist (idx=0) before feeding model")
    ap.add_argument("--control", type=str, default="infer", choices=["infer", "gt"], help="which action drives the sim")
    ap.add_argument("--print_every", type=int, default=30, help="print compare metrics every N frames")
    ap.add_argument("--clamp_min", type=float, default=-1.57)
    ap.add_argument("--clamp_max", type=float, default=1.57)
    ap.add_argument("--max_delta", type=float, default=0.08, help="per-step max delta (radians); 0 disables")
    args = ap.parse_args()

    data_path = str(args.data_path)
    gt_action20 = None
    if data_path.endswith(".npz"):
        tips5, gt_action20 = _load_wuji_npz(data_path)
        T = int(tips5.shape[0])
    else:
        data = _load_pts21_npy(data_path)
        T = int(data.shape[0])
    start, end = _slice_range(T, args.start, args.end)

    # Load GeoRT model
    sys.path.insert(0, str((_repo_root() / "wuji_retarget").resolve()))
    import geort  # type: ignore

    policy = geort.load_model(args.ckpt_tag, epoch=int(args.epoch))
    try:
        policy.eval()
    except Exception:
        pass

    # Load MuJoCo
    import mujoco  # type: ignore
    import mujoco.viewer  # type: ignore

    mjcf = args.mjcf_path.strip() or _default_mjcf(args.hand_side)
    if not Path(mjcf).exists():
        raise FileNotFoundError(mjcf)

    model = mujoco.MjModel.from_xml_path(mjcf)
    mjdata = mujoco.MjData(model)

    # init ctrl to mid-range (same as wuji_retargeting sim)
    for i in range(int(model.nu)):
        if bool(model.actuator_ctrllimited[i]):
            lo, hi = model.actuator_ctrlrange[i]
            mjdata.ctrl[i] = 0.5 * (float(lo) + float(hi))
        else:
            mjdata.ctrl[i] = 0.0
    for _ in range(50):
        mujoco.mj_step(model, mjdata)

    dt = 1.0 / max(1.0, float(args.fps))
    sim_dt = float(model.opt.timestep)
    steps_per_ctrl = max(1, int(round(dt / max(1e-6, sim_dt))))

    last_qpos: Optional[np.ndarray] = None
    frame_count = 0
    t = start

    print("=" * 70)
    if gt_action20 is not None:
        print(f"[replay] data: {args.data_path}  tips5={tips5.shape}  gt_action20={gt_action20.shape}  range=[{start},{end}) loop={not args.no_loop}")
    else:
        print(f"[replay] data: {args.data_path}  pts21={data.shape}  range=[{start},{end}) loop={not args.no_loop}")
    print(f"[replay] ckpt: tag={args.ckpt_tag} epoch={args.epoch}")
    print(f"[replay] mjcf: {mjcf}  nu={model.nu}  ctrl_dt={dt:.4f}s steps_per_ctrl={steps_per_ctrl}")
    print(f"[replay] control: {args.control}  print_every={args.print_every}")
    print("=" * 70)

    with mujoco.viewer.launch_passive(model, mjdata) as viewer:
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0, 0, 0.05]

        while viewer.is_running():
            loop_t0 = time.time()

            if gt_action20 is not None:
                human_points5 = tips5[t].astype(np.float32).reshape(5, 3)  # already rel_wrist
                gt20 = gt_action20[t].astype(np.float32).reshape(20)
            else:
                pts21 = data[t]
                human_points5 = _points5_from_pts21(pts21, relative_to_wrist=bool(args.relative_to_wrist))
                gt20 = None

            infer20 = policy.forward(human_points5)  # expect (20,)
            infer20 = np.asarray(infer20, dtype=np.float32).reshape(-1)
            if infer20.shape[0] != 20:
                raise ValueError(f"Policy output dim mismatch: expect 20, got {infer20.shape}")

            # safety: clamp + optional rate limit
            infer20 = np.clip(infer20, float(args.clamp_min), float(args.clamp_max))
            if last_qpos is not None and float(args.max_delta) > 0:
                delta = np.clip(infer20 - last_qpos, -float(args.max_delta), float(args.max_delta))
                infer20 = last_qpos + delta
            last_qpos = infer20.copy()

            if gt20 is not None and (frame_count % max(1, int(args.print_every)) == 0):
                diff = infer20 - gt20
                l2 = float(np.linalg.norm(diff))
                linf = float(np.max(np.abs(diff)))
                print(f"[cmp] t={t}  ||infer-gt||2={l2:.4f}  ||.||inf={linf:.4f}", flush=True)

            drive = infer20 if str(args.control).lower() == "infer" else (gt20 if gt20 is not None else infer20)
            mjdata.ctrl[: min(int(model.nu), 20)] = drive[: min(int(model.nu), 20)]

            for _ in range(steps_per_ctrl):
                mujoco.mj_step(model, mjdata)
            viewer.sync()

            # advance frame (with speed multiplier)
            step = max(1, int(round(float(args.speed))))
            t += step
            if t >= end:
                if args.no_loop:
                    break
                t = start

            # rate limit
            elapsed = time.time() - loop_t0
            sleep_s = max(0.0, dt - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)
            frame_count += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


