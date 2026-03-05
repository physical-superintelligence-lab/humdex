from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


@dataclass(frozen=True)
class VdhandConfig:
    dst_ip: str
    dst_port: int
    mocap_index: int = 0
    local_port: int = 0
    hands: str = "both"


class VdhandReader:
    def __init__(self, cfg: VdhandConfig) -> None:
        self.cfg = cfg
        self.sdk: Optional[Dict[str, Any]] = None
        self.mocap_data: Any = None
        self.length_hand: int = 0
        self.joint_names_lhand: list[str] = []
        self.joint_names_rhand: list[str] = []
        self._initialized = False

    def _import_sdk(self) -> Dict[str, Any]:
        sdk_dir = _repo_root() / "vdmocap" / "DataRead_Python_Demo"
        if not sdk_dir.exists():
            raise FileNotFoundError(f"Cannot find VDMocap SDK path: {sdk_dir}")
        if str(sdk_dir) not in sys.path:
            sys.path.insert(0, str(sdk_dir))
        from vdmocapsdk_dataread import (  # type: ignore
            MocapData,
            udp_close,
            udp_is_open,
            udp_open,
            udp_recv_mocap_data,
            udp_remove,
            udp_send_request_connect,
        )
        from vdmocapsdk_nodelist import LENGTH_HAND, NAMES_JOINT_HAND_LEFT, NAMES_JOINT_HAND_RIGHT  # type: ignore

        return {
            "MocapData": MocapData,
            "udp_open": udp_open,
            "udp_is_open": udp_is_open,
            "udp_close": udp_close,
            "udp_remove": udp_remove,
            "udp_send_request_connect": udp_send_request_connect,
            "udp_recv_mocap_data": udp_recv_mocap_data,
            "LENGTH_HAND": int(LENGTH_HAND),
            "NAMES_JOINT_HAND_LEFT": list(NAMES_JOINT_HAND_LEFT),
            "NAMES_JOINT_HAND_RIGHT": list(NAMES_JOINT_HAND_RIGHT),
        }

    def initialize(self) -> None:
        self.sdk = self._import_sdk()
        self.mocap_data = self.sdk["MocapData"]()
        self.length_hand = int(self.sdk["LENGTH_HAND"])
        self.joint_names_lhand = [str(x) for x in self.sdk["NAMES_JOINT_HAND_LEFT"]]
        self.joint_names_rhand = [str(x) for x in self.sdk["NAMES_JOINT_HAND_RIGHT"]]

        idx = int(self.cfg.mocap_index)
        if not bool(self.sdk["udp_is_open"](idx)):
            ok = bool(self.sdk["udp_open"](idx, int(self.cfg.local_port)))
            if not ok:
                raise RuntimeError(f"Failed to open VDHand UDP (index={idx})")
        ok = bool(self.sdk["udp_send_request_connect"](idx, str(self.cfg.dst_ip), int(self.cfg.dst_port)))
        if not ok:
            raise RuntimeError(f"Failed to connect VDHand sender: {self.cfg.dst_ip}:{self.cfg.dst_port}")
        self._initialized = True

    def read_frame(self) -> Dict[str, Any]:
        if not self._initialized or self.sdk is None or self.mocap_data is None:
            return {"ok": False, "reason": "not_initialized"}

        got = bool(
            self.sdk["udp_recv_mocap_data"](
                int(self.cfg.mocap_index), str(self.cfg.dst_ip), int(self.cfg.dst_port), self.mocap_data
            )
        )
        if (not got) or (not bool(getattr(self.mocap_data, "isUpdate", False))):
            return {"ok": False, "reason": "no_update"}

        fi = int(getattr(self.mocap_data, "frameIndex", -1))
        hands = str(self.cfg.hands).lower()
        fr: Dict[str, Any] = {}
        if hands in ["left", "both"]:
            pos_arr = self.mocap_data.position_lHand
            quat_arr = self.mocap_data.quaternion_lHand
            for i in range(self.length_hand):
                name = str(self.joint_names_lhand[i])
                p = np.array([float(pos_arr[i][0]), float(pos_arr[i][1]), float(pos_arr[i][2])], dtype=np.float32)
                q = np.array([float(quat_arr[i][0]), float(quat_arr[i][1]), float(quat_arr[i][2]), float(quat_arr[i][3])], dtype=np.float32)
                fr[name] = [p, _safe_quat_wxyz(q)]
        if hands in ["right", "both"]:
            pos_arr = self.mocap_data.position_rHand
            quat_arr = self.mocap_data.quaternion_rHand
            for i in range(self.length_hand):
                name = str(self.joint_names_rhand[i])
                p = np.array([float(pos_arr[i][0]), float(pos_arr[i][1]), float(pos_arr[i][2])], dtype=np.float32)
                q = np.array([float(quat_arr[i][0]), float(quat_arr[i][1]), float(quat_arr[i][2]), float(quat_arr[i][3])], dtype=np.float32)
                fr[name] = [p, _safe_quat_wxyz(q)]

        return {"ok": True, "frame_index": fi, "hand_frame": fr}

    def close(self) -> None:
        if self.sdk is None:
            return
        idx = int(self.cfg.mocap_index)
        try:
            self.sdk["udp_close"](idx)
        except Exception:
            pass
        try:
            self.sdk["udp_remove"](idx)
        except Exception:
            pass
        self._initialized = False


@dataclass(frozen=True)
class VdhandRuntimeConfig:
    hand_fk: bool
    hand_fk_end_site_scale: str
    hand_no_csv_transform: bool
    csv_geo_to_bvh_official: bool
    csv_apply_bvh_rotation: bool
    hands: str


def build_vdhand_tracking(
    *,
    hands_mode: str,
    is_active: bool,
    now_ms: int,
    fr_hand: Optional[Dict[str, Any]],
    cfg_hand_fk: bool,
    runtime: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    ht_l: Dict[str, Any] = {"is_active": bool(is_active and hands_mode in ["left", "both"]), "timestamp": now_ms}
    ht_r: Dict[str, Any] = {"is_active": bool(is_active and hands_mode in ["right", "both"]), "timestamp": now_ms}
    if (not isinstance(fr_hand, dict)) or (not bool(is_active)):
        return ht_l, ht_r

    mapper = runtime.get("_hand_to_tracking26", None)
    if mapper is None:
        return ht_l, ht_r

    fk_left = runtime.get("fk_bone_left", None)
    fk_right = runtime.get("fk_bone_right", None)
    hand_fk_scale = runtime.get("hand_fk_end_site_scale", 0.8)
    name_fn = runtime.get("_hand_joint_order_names", None)
    fk_fn = runtime.get("_fk_hand_positions_with_end_sites", None)
    safe_quat = runtime.get("_safe_quat_wxyz", None)

    if hands_mode in ["left", "both"]:
        try:
            pos_override = None
            tip_override = None
            if bool(cfg_hand_fk) and (fk_left is not None) and callable(name_fn) and callable(fk_fn) and callable(safe_quat):
                names = name_fn("Left")
                q20 = []
                for n in names:
                    v = fr_hand.get(n, None)
                    if not (isinstance(v, (list, tuple)) and len(v) >= 2):
                        raise KeyError(f"missing joint in frame: {n}")
                    q20.append(safe_quat(np.asarray(v[1], dtype=np.float32)))
                q20 = np.stack(q20, axis=0)
                root_pos = np.asarray(fr_hand["LeftHand"][0], dtype=np.float32).reshape(3)
                pos20, pos_end5 = fk_fn(
                    q20,
                    root_pos=root_pos,
                    bone_init_pos=fk_left,
                    end_site_scale=hand_fk_scale,
                )
                pos_override = {n: pos20[i] for i, n in enumerate(names)}
                tip_override = {
                    "LeftHandThumbTip": pos_end5[0],
                    "LeftHandIndexTip": pos_end5[1],
                    "LeftHandMiddleTip": pos_end5[2],
                    "LeftHandRingTip": pos_end5[3],
                    "LeftHandLittleTip": pos_end5[4],
                }
            ht_l.update(mapper(fr_hand, "left", pos_override=pos_override, tip_override=tip_override))
        except Exception:
            pass

    if hands_mode in ["right", "both"]:
        try:
            pos_override = None
            tip_override = None
            if bool(cfg_hand_fk) and (fk_right is not None) and callable(name_fn) and callable(fk_fn) and callable(safe_quat):
                names = name_fn("Right")
                q20 = []
                for n in names:
                    v = fr_hand.get(n, None)
                    if not (isinstance(v, (list, tuple)) and len(v) >= 2):
                        raise KeyError(f"missing joint in frame: {n}")
                    q20.append(safe_quat(np.asarray(v[1], dtype=np.float32)))
                q20 = np.stack(q20, axis=0)
                root_pos = np.asarray(fr_hand["RightHand"][0], dtype=np.float32).reshape(3)
                pos20, pos_end5 = fk_fn(
                    q20,
                    root_pos=root_pos,
                    bone_init_pos=fk_right,
                    end_site_scale=hand_fk_scale,
                )
                pos_override = {n: pos20[i] for i, n in enumerate(names)}
                tip_override = {
                    "RightHandThumbTip": pos_end5[0],
                    "RightHandIndexTip": pos_end5[1],
                    "RightHandMiddleTip": pos_end5[2],
                    "RightHandRingTip": pos_end5[3],
                    "RightHandLittleTip": pos_end5[4],
                }
            ht_r.update(mapper(fr_hand, "right", pos_override=pos_override, tip_override=tip_override))
        except Exception:
            pass
    return ht_l, ht_r


def build_vdhand_runtime(cfg: VdhandRuntimeConfig) -> Dict[str, Any]:
    from deploy_real.pose_csv_loader import (  # type: ignore
        apply_bvh_like_coordinate_transform,
        apply_geo_to_bvh_official,
    )
    from deploy_real.common.teleop_compat import (
        INITIAL_POSITION_HAND_LEFT,
        INITIAL_POSITION_HAND_RIGHT,
        _parse_hand_fk_end_site_scale,
        _hand_to_tracking26,
        _hand_joint_order_names,
        _fk_hand_positions_with_end_sites,
        _safe_quat_wxyz,
    )

    runtime: Dict[str, Any] = {
        "_hand_to_tracking26": _hand_to_tracking26,
        "_hand_joint_order_names": _hand_joint_order_names,
        "_fk_hand_positions_with_end_sites": _fk_hand_positions_with_end_sites,
        "_safe_quat_wxyz": _safe_quat_wxyz,
        "build_hand_tracking": build_vdhand_tracking,
        "build_vdhand_tracking": build_vdhand_tracking,
        "hand_fk_end_site_scale": _parse_hand_fk_end_site_scale(str(cfg.hand_fk_end_site_scale)),
        "fk_bone_left": None,
        "fk_bone_right": None,
    }

    if bool(cfg.hand_fk):
        try:
            names_l = _hand_joint_order_names("Left")
            names_r = _hand_joint_order_names("Right")
            fr_l = {}
            fr_r = {}
            for n, p0 in zip(names_l, INITIAL_POSITION_HAND_LEFT):
                fr_l[n] = [np.asarray(p0, dtype=np.float32).reshape(3), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
            for n, p0 in zip(names_r, INITIAL_POSITION_HAND_RIGHT):
                fr_r[n] = [np.asarray(p0, dtype=np.float32).reshape(3), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
            if (not bool(cfg.hand_no_csv_transform)) and bool(cfg.csv_geo_to_bvh_official):
                fr_l = apply_geo_to_bvh_official(fr_l)
                fr_r = apply_geo_to_bvh_official(fr_r)
            if (not bool(cfg.hand_no_csv_transform)) and bool(cfg.csv_apply_bvh_rotation):
                fr_l = apply_bvh_like_coordinate_transform(fr_l, pos_unit="m", apply_rotation=True)
                fr_r = apply_bvh_like_coordinate_transform(fr_r, pos_unit="m", apply_rotation=True)
            runtime["fk_bone_left"] = np.stack([np.asarray(fr_l[n][0], dtype=np.float32).reshape(3) for n in names_l], axis=0)
            runtime["fk_bone_right"] = np.stack([np.asarray(fr_r[n][0], dtype=np.float32).reshape(3) for n in names_r], axis=0)
        except Exception:
            runtime["fk_bone_left"] = None
            runtime["fk_bone_right"] = None

    return runtime

