#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click collector for GeoRT supervised training data.

What it does:
- Scans the two SRC_DIRS for "*left.npz" and "*left_*.npz" files (same rule as your original).
- For each file, reads:
    HUMAN_KEY = "fingertips_rel_wrist"  (expects [T,5,3] or [T,15] -> reshaped to [T,5,3])
    ROBOT_IN_KEY = "action_wuji_qpos_target" (expects [T,DOF] or [DOF] -> reshaped to [1,DOF])
- Truncates per-file to min(T_human, T_robot) for alignment.
- Concats all segments and saves ONE output NPZ.

Critical fix for your training error:
- Writes supervision joint key "qpos" (and a few aliases) so SupervisedRetargetDataset can find it.

Run:
    python collect.py
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# ===================== DEFAULT CONFIG =====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "wuji_policy" / "data"

# Required keys in source files
HUMAN_KEY = "fingertips_rel_wrist"
ROBOT_IN_KEY = "action_wuji_qpos_target"

# Output supervision key expected by SupervisedRetargetDataset
QPOS_KEY_OUT = "qpos"

# Optional aliases (harmless; increases compatibility)
QPOS_ALIAS_KEYS = ["robot_qpos", "joint", "joint_angle", "joint_angles"]

# Keep the original robot key as well (nice for debugging)
KEEP_ORIGINAL_ROBOT_KEY = True

# Meta keys: keep the first occurrence if present
META_KEYS = [
    "fingertip_names",
    "hand_side",
    "mediapipe_21d_keypoint_names",
    "source_data_json",
]

# Optional time-aligned keys: ONLY saved if present in ALL kept files
OPTIONAL_CONCAT_KEYS = [
    "frame_idx",
    "timestamp_ms",
    "mediapipe_21d_transformed",
]

# =======================================================================


def _npz_load(path: Path):
    return np.load(path, allow_pickle=True)


def _as_array(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _normalize_human(arr: np.ndarray) -> np.ndarray:
    """
    Accept:
      - [T,5,3]
      - [T,15] -> reshape to [T,5,3]
    """
    arr = _as_array(arr)
    if arr.ndim == 3 and arr.shape[1:] == (5, 3):
        return arr
    if arr.ndim == 2 and arr.shape[1] == 15:
        return arr.reshape(arr.shape[0], 5, 3)
    raise ValueError(f"{HUMAN_KEY} unexpected shape {arr.shape} (expect [T,5,3] or [T,15])")


def _normalize_robot(arr: np.ndarray) -> np.ndarray:
    """
    Accept:
      - [T,DOF]
      - [DOF] -> [1,DOF]
    """
    arr = _as_array(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f"{ROBOT_IN_KEY} unexpected shape {arr.shape} (expect [T,DOF] or [DOF])")


def _collect_files(input_path: Path):
    if not input_path.exists():
        print(f"[ERROR] input_path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            print(f"[ERROR] input_path must be a .npz file when passing a file: {input_path}", file=sys.stderr)
            sys.exit(1)
        return [input_path.resolve()]

    # directory mode: collect all npz recursively
    files = sorted([p.resolve() for p in input_path.rglob("*.npz") if p.is_file()], key=lambda p: str(p))
    return files


def _resolve_output_path(input_path: Path, output_name):
    if output_name:
        name = output_name if output_name.endswith(".npz") else f"{output_name}.npz"
    else:
        name = f"{input_path.stem}.npz" if input_path.is_file() else f"{input_path.name}.npz"
    return (DEFAULT_OUT_DIR / name).resolve()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Wuji supervised training data and save one merged npz to ../wuji_policy/data"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input .npz file or a directory containing multiple .npz files",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Optional output file name (with or without .npz). Default derives from input_path name.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    out_path = _resolve_output_path(input_path, args.output_name.strip() or None)

    files = _collect_files(input_path)
    if not files:
        print(f"[ERROR] No npz files found in: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(files)} npz files from input: {input_path}")
    for p in files[:10]:
        print("  ", p)
    if len(files) > 10:
        print("  ...")

    human_chunks = []
    robot_chunks = []

    # optional chunks: collect per key
    optional_chunks = {k: [] for k in OPTIONAL_CONCAT_KEYS}
    optional_presence = {k: 0 for k in OPTIONAL_CONCAT_KEYS}  # count files that contain it

    meta_store = {}

    kept = 0
    skipped = 0

    for fp in files:
        z = _npz_load(fp)
        keys = set(z.files)

        # required check
        missing = [k for k in (HUMAN_KEY, ROBOT_IN_KEY) if k not in keys]
        if missing:
            print(f"[WARN] Skip {fp.name}: missing keys {missing}. Available keys={z.files}")
            z.close()
            skipped += 1
            continue

        try:
            human = _normalize_human(z[HUMAN_KEY])
            robot = _normalize_robot(z[ROBOT_IN_KEY])

            # align by truncation
            T = min(human.shape[0], robot.shape[0])
            if T <= 0:
                raise ValueError("T <= 0 after alignment")

            if human.shape[0] != robot.shape[0]:
                print(f"[WARN] {fp.name}: length mismatch {HUMAN_KEY}.T={human.shape[0]} vs {ROBOT_IN_KEY}.T={robot.shape[0]} -> truncate to {T}")

            human = human[:T]
            robot = robot[:T]

            human_chunks.append(human.astype(np.float32))
            robot_chunks.append(robot.astype(np.float32))

            # optional keys: keep only if can be time-aligned
            for k in OPTIONAL_CONCAT_KEYS:
                if k in keys:
                    arr = _as_array(z[k])
                    if arr.ndim >= 1 and arr.shape[0] >= T:
                        optional_chunks[k].append(arr[:T])
                        optional_presence[k] += 1
                    else:
                        # present but unusable -> treat as missing for this file
                        print(f"[WARN] {fp.name}: optional '{k}' has shape {arr.shape}, cannot align to T={T}; will skip saving '{k}' globally.")

            # meta: keep first
            for k in META_KEYS:
                if k in keys and k not in meta_store:
                    meta_store[k] = z[k]

            kept += 1

        except Exception as e:
            print(f"[WARN] Skip {fp.name}: {type(e).__name__}: {e}")
            skipped += 1
        finally:
            z.close()

    if kept == 0:
        print("[ERROR] No usable files after filtering. Please check source files contain both required keys.", file=sys.stderr)
        sys.exit(1)

    human_all = np.concatenate(human_chunks, axis=0)
    robot_all = np.concatenate(robot_chunks, axis=0)

    # Build output dict
    out = {
        HUMAN_KEY: human_all,                 # [T,5,3]
        QPOS_KEY_OUT: robot_all,              # [T,DOF]  <-- critical for SupervisedRetargetDataset
    }

    # Add aliases for robustness/compat
    for k in QPOS_ALIAS_KEYS:
        if k and k not in out:
            out[k] = robot_all

    # Optionally keep original key
    if KEEP_ORIGINAL_ROBOT_KEY:
        out[ROBOT_IN_KEY] = robot_all

    # Optional concat keys: ONLY save if present in ALL kept files
    total_T = human_all.shape[0]
    for k in OPTIONAL_CONCAT_KEYS:
        if optional_presence[k] != kept:
            if optional_presence[k] > 0:
                print(f"[INFO] Optional key '{k}' present in {optional_presence[k]}/{kept} kept files -> NOT saving to avoid misalignment.")
            continue
        try:
            arr_all = np.concatenate(optional_chunks[k], axis=0)
            if arr_all.shape[0] != total_T:
                print(f"[WARN] Optional key '{k}' total length {arr_all.shape[0]} != total_T {total_T} -> NOT saving.")
                continue
            out[k] = arr_all
        except Exception as e:
            print(f"[WARN] Cannot concat optional key '{k}': {type(e).__name__}: {e}. NOT saving.")

    # Meta keys
    for k, v in meta_store.items():
        out[k] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    print(f"[INFO] Saved: {out_path}")
    print(f"[INFO] kept={kept}, skipped={skipped}, total_T={human_all.shape[0]}")
    print("[INFO] Saved keys & shapes:")
    for k in sorted(out.keys()):
        v = out[k]
        if isinstance(v, np.ndarray):
            print(f"  {k:28s} shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k:28s} type={type(v)}")

    # Training sanity checks
    print("[CHECK]", HUMAN_KEY, "OK" if HUMAN_KEY in out else "MISSING")
    print("[CHECK]", QPOS_KEY_OUT, "OK" if QPOS_KEY_OUT in out else "MISSING")


if __name__ == "__main__":
    main()
