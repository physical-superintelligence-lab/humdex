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
class VdmocapBodyConfig:
    dst_ip: str
    dst_port: int
    mocap_index: int = 0
    local_port: int = 0
    world_space: int = 0


class VdmocapBodyReader:
    def __init__(self, cfg: VdmocapBodyConfig) -> None:
        self.cfg = cfg
        self.sdk: Optional[Dict[str, Any]] = None
        self.mocap_data: Any = None
        self.joint_names: list[str] = []
        self.length_body: int = 0
        self._initialized = False
        self.length_hand: int = 0
        self.joint_names_lhand: list[str] = []
        self.joint_names_rhand: list[str] = []
        self._last_frame_index: int = -1

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
            udp_set_position_in_initial_tpose,
        )
        from vdmocapsdk_nodelist import (  # type: ignore
            LENGTH_BODY,
            LENGTH_HAND,
            NAMES_JOINT_BODY,
            NAMES_JOINT_HAND_LEFT,
            NAMES_JOINT_HAND_RIGHT,
        )

        return {
            "MocapData": MocapData,
            "udp_open": udp_open,
            "udp_is_open": udp_is_open,
            "udp_close": udp_close,
            "udp_remove": udp_remove,
            "udp_send_request_connect": udp_send_request_connect,
            "udp_set_position_in_initial_tpose": udp_set_position_in_initial_tpose,
            "udp_recv_mocap_data": udp_recv_mocap_data,
            "LENGTH_BODY": int(LENGTH_BODY),
            "LENGTH_HAND": int(LENGTH_HAND),
            "NAMES_JOINT_BODY": list(NAMES_JOINT_BODY),
            "NAMES_JOINT_HAND_LEFT": list(NAMES_JOINT_HAND_LEFT),
            "NAMES_JOINT_HAND_RIGHT": list(NAMES_JOINT_HAND_RIGHT),
        }

    def initialize(self) -> None:
        self.sdk = self._import_sdk()
        self.mocap_data = self.sdk["MocapData"]()
        self.length_body = int(self.sdk["LENGTH_BODY"])
        self.joint_names = [str(x) for x in self.sdk["NAMES_JOINT_BODY"]]
        self.length_hand = int(self.sdk["LENGTH_HAND"])
        self.joint_names_lhand = [str(x) for x in self.sdk["NAMES_JOINT_HAND_LEFT"]]
        self.joint_names_rhand = [str(x) for x in self.sdk["NAMES_JOINT_HAND_RIGHT"]]

        idx = int(self.cfg.mocap_index)
        if not bool(self.sdk["udp_is_open"](idx)):
            ok = bool(self.sdk["udp_open"](idx, int(self.cfg.local_port)))
            if not ok:
                raise RuntimeError(f"Failed to open VDMocap UDP (index={idx})")
        from deploy_real.common.teleop_compat import INITIAL_POSITION_BODY, INITIAL_POSITION_HAND_LEFT, INITIAL_POSITION_HAND_RIGHT
        try:
            self.sdk["udp_set_position_in_initial_tpose"](
                idx,
                str(self.cfg.dst_ip),
                int(self.cfg.dst_port),
                int(self.cfg.world_space),
                INITIAL_POSITION_BODY,
                INITIAL_POSITION_HAND_RIGHT,
                INITIAL_POSITION_HAND_LEFT,
            )
        except Exception:
            # Not fatal for reader-side initialization.
            pass
        ok = bool(self.sdk["udp_send_request_connect"](idx, str(self.cfg.dst_ip), int(self.cfg.dst_port)))
        if not ok:
            raise RuntimeError(f"Failed to connect VDMocap sender: {self.cfg.dst_ip}:{self.cfg.dst_port}")
        self._last_frame_index = -1
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
        # Match legacy teleop behavior: process only fresh mocap frames.
        # Re-consuming the same frame can make IK bounce on lower-body joints.
        if fi == self._last_frame_index:
            return {"ok": False, "reason": "no_update"}
        self._last_frame_index = fi
        fr: Dict[str, Any] = {}
        for i in range(self.length_body):
            name = self.joint_names[i]
            p = np.array(
                [
                    float(self.mocap_data.position_body[i][0]),
                    float(self.mocap_data.position_body[i][1]),
                    float(self.mocap_data.position_body[i][2]),
                ],
                dtype=np.float32,
            )
            q = np.array(
                [
                    float(self.mocap_data.quaternion_body[i][0]),
                    float(self.mocap_data.quaternion_body[i][1]),
                    float(self.mocap_data.quaternion_body[i][2]),
                    float(self.mocap_data.quaternion_body[i][3]),
                ],
                dtype=np.float32,
            )
            fr[name] = [p, _safe_quat_wxyz(q)]

        fr_hand: Dict[str, Any] = {}
        for i in range(self.length_hand):
            name_l = str(self.joint_names_lhand[i])
            pl = np.array(
                [
                    float(self.mocap_data.position_lHand[i][0]),
                    float(self.mocap_data.position_lHand[i][1]),
                    float(self.mocap_data.position_lHand[i][2]),
                ],
                dtype=np.float32,
            )
            ql = np.array(
                [
                    float(self.mocap_data.quaternion_lHand[i][0]),
                    float(self.mocap_data.quaternion_lHand[i][1]),
                    float(self.mocap_data.quaternion_lHand[i][2]),
                    float(self.mocap_data.quaternion_lHand[i][3]),
                ],
                dtype=np.float32,
            )
            fr_hand[name_l] = [pl, _safe_quat_wxyz(ql)]
        for i in range(self.length_hand):
            name_r = str(self.joint_names_rhand[i])
            pr = np.array(
                [
                    float(self.mocap_data.position_rHand[i][0]),
                    float(self.mocap_data.position_rHand[i][1]),
                    float(self.mocap_data.position_rHand[i][2]),
                ],
                dtype=np.float32,
            )
            qr = np.array(
                [
                    float(self.mocap_data.quaternion_rHand[i][0]),
                    float(self.mocap_data.quaternion_rHand[i][1]),
                    float(self.mocap_data.quaternion_rHand[i][2]),
                    float(self.mocap_data.quaternion_rHand[i][3]),
                ],
                dtype=np.float32,
            )
            fr_hand[name_r] = [pr, _safe_quat_wxyz(qr)]

        return {"ok": True, "frame_index": fi, "body_frame": fr, "hand_frame": fr_hand}

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
class VdmocapRuntimeConfig:
    format: str
    actual_human_height: float
    smooth: bool
    smooth_window_size: int
    safe_idle_pose_id: str


def build_vdmocap_runtime(cfg: VdmocapRuntimeConfig) -> Dict[str, Any]:
    from deploy_real.pose_csv_loader import (  # type: ignore
        apply_bvh_like_coordinate_transform,
        apply_geo_to_bvh_official,
        gmr_rename_and_footmod,
    )
    from general_motion_retargeting import GeneralMotionRetargeting as GMR  # type: ignore
    from general_motion_retargeting import human_head_to_robot_neck  # type: ignore
    from deploy_real.common.teleop_compat import (
        SAFE_IDLE_BODY_35_PRESETS,
        _parse_safe_idle_pose_ids,
        extract_mimic_obs_whole_body,
        SmoothFilter,
    )

    retargeter = GMR(
        src_human=f"bvh_{cfg.format}",
        tgt_robot="unitree_g1",
        actual_human_height=float(cfg.actual_human_height),
    )
    smooth_filter = SmoothFilter(enable=bool(cfg.smooth), window_size=max(1, int(cfg.smooth_window_size)))

    safe_ids = _parse_safe_idle_pose_ids(cfg.safe_idle_pose_id)
    safe_seq = []
    if len(safe_ids) > 0:
        for pid in safe_ids:
            safe_seq.append(list(SAFE_IDLE_BODY_35_PRESETS[int(pid)]))

    return {
        "retargeter": retargeter,
        "apply_geo_to_bvh_official": apply_geo_to_bvh_official,
        "apply_bvh_like_coordinate_transform": apply_bvh_like_coordinate_transform,
        "gmr_rename_and_footmod": gmr_rename_and_footmod,
        "extract_mimic_obs_whole_body": extract_mimic_obs_whole_body,
        "human_head_to_robot_neck": human_head_to_robot_neck,
        "smooth_filter": smooth_filter,
        "safe_idle_body_seq_35": safe_seq,
    }

