from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _safe_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if (not np.isfinite(n)) or n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _axis_swap_flip_matrix(swap: str, flip: str) -> np.ndarray:
    swap = str(swap).strip().lower()
    flip = str(flip).strip().lower()
    axes = {"x": 0, "y": 1, "z": 2}
    if len(swap) != 3 or any(c not in axes for c in swap):
        swap = "xyz"
    idx = [axes[c] for c in swap]
    m = np.eye(3, dtype=np.float32)[:, idx]
    f = np.ones(3, dtype=np.float32)
    for c in flip:
        if c in axes:
            f[axes[c]] *= -1.0
    return m * f.reshape(1, 3)


def _decode_type02_item_candidates(
    v1: float, v2: float, v3: float, v4: float, v5: float, v6: float, v7: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    # Candidate A: [x, y, z, qw, qx, qy, qz]
    q_a_raw = np.array([v4, v5, v6, v7], dtype=np.float32)
    p_a = np.array([v1, v2, v3], dtype=np.float32)
    score_a = abs(float(np.linalg.norm(q_a_raw)) - 1.0)

    # Candidate B: [qw, qx, qy, qz, x, y, z]
    q_b_raw = np.array([v1, v2, v3, v4], dtype=np.float32)
    p_b = np.array([v5, v6, v7], dtype=np.float32)
    score_b = abs(float(np.linalg.norm(q_b_raw)) - 1.0)

    return p_a, q_a_raw, score_a, p_b, q_b_raw, score_b


@dataclass(frozen=True)
class XsensBodyConfig:
    listen_ip: str = "0.0.0.0"
    listen_port: int = 9763
    socket_timeout_s: float = 0.05
    character_id_filter: int = -1
    drop_incomplete_samples: bool = True
    axis_swap: str = "xyz"  # legacy no-op
    axis_flip: str = ""  # legacy no-op
    use_bvh_fk: bool = False  # legacy no-op
    use_viewer_fk: bool = True  # legacy no-op
    fk_rot_mode: str = "global"  # legacy no-op
    bvh_path: str = "assets/slimevr/bvh-recording.bvh"  # legacy no-op
    bvh_scale: float = 0.01  # legacy no-op
    bone_axis_override: str = ""  # legacy no-op
    swap_lr_legs: bool = False  # legacy no-op
    quat_order: str = "auto"  # auto|wxyz|xyzw


@dataclass
class _SampleState:
    segments: Dict[int, list[np.ndarray]]
    got_last: bool = False
    max_part_index: int = -1
    updated_ts: float = 0.0


class XsensBodyReader:
    _HEADER = struct.Struct(">6sIBBIBBBBHH")
    _ITEM = struct.Struct(">I7f")
    _MAX_PENDING = 16
    _STALE_SECONDS = 1.5
    _SAMPLE_RESET_THRESHOLD = 2000

    # Map MVN segment IDs to names matching our existing VDMocap/GMR flow.
    _BODY_ID_TO_NAME = {
        1: "Hips",
        2: "Spine",
        3: "Spine1",
        4: "Spine2",
        5: "Spine3",
        6: "Neck",
        7: "Head",
        8: "RightShoulder",
        9: "RightUpperArm",
        10: "RightLowerArm",
        11: "RightHand",
        12: "LeftShoulder",
        13: "LeftUpperArm",
        14: "LeftLowerArm",
        15: "LeftHand",
        16: "RightUpperLeg",
        17: "RightLowerLeg",
        18: "RightFoot",
        19: "RightToe",
        20: "LeftUpperLeg",
        21: "LeftLowerLeg",
        22: "LeftFoot",
        23: "LeftToe",
    }
    _XSENS_TO_GMR_NAME = {
        "Hips": "Pelvis",
        "Spine": "Spine",
        "Spine1": "Spine1",
        "Spine2": "Spine2",
        "Spine3": "Chest",
        "Neck": "Neck",
        "Head": "Head",
        "LeftShoulder": "Left_Shoulder",
        "LeftUpperArm": "Left_UpperArm",
        "LeftLowerArm": "Left_Forearm",
        "LeftHand": "Left_Hand",
        "RightShoulder": "Right_Shoulder",
        "RightUpperArm": "Right_UpperArm",
        "RightLowerArm": "Right_Forearm",
        "RightHand": "Right_Hand",
        "LeftUpperLeg": "Left_UpperLeg",
        "LeftLowerLeg": "Left_LowerLeg",
        "LeftFoot": "Left_Foot",
        "LeftToe": "Left_Toe",
        "RightUpperLeg": "Right_UpperLeg",
        "RightLowerLeg": "Right_LowerLeg",
        "RightFoot": "Right_Foot",
        "RightToe": "Right_Toe",
    }

    def __init__(self, cfg: XsensBodyConfig) -> None:
        self.cfg = cfg
        self.sock: Optional[socket.socket] = None
        self._initialized = False
        self._pending: Dict[int, _SampleState] = {}
        self._last_frame_index = -1
        self._position_scale_to_m: Optional[float] = None
        self._type02_layout: Optional[str] = None  # "xyz_q" or "q_xyz"
        self._quat_order = str(cfg.quat_order).strip().lower()
        if self._quat_order not in ("auto", "wxyz", "xyzw"):
            self._quat_order = "auto"
        self._last_quat_by_seg: Dict[int, np.ndarray] = {}
        self._gmr_initial_yaw_inv_wxyz: Optional[np.ndarray] = None

    def initialize(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((str(self.cfg.listen_ip), int(self.cfg.listen_port)))
        sock.settimeout(max(0.005, float(self.cfg.socket_timeout_s)))
        self.sock = sock
        self._pending = {}
        self._last_frame_index = -1
        self._type02_layout = None
        self._position_scale_to_m = None
        self._last_quat_by_seg = {}
        self._gmr_initial_yaw_inv_wxyz = None
        self._initialized = True

    @staticmethod
    def _quat_mul_wxyz(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
        a = np.asarray(qa, dtype=np.float32).reshape(4)
        b = np.asarray(qb, dtype=np.float32).reshape(4)
        wa, xa, ya, za = [float(v) for v in a]
        wb, xb, yb, zb = [float(v) for v in b]
        out = np.array(
            [
                wa * wb - xa * xb - ya * yb - za * zb,
                wa * xb + xa * wb + ya * zb - za * yb,
                wa * yb - xa * zb + ya * wb + za * xb,
                wa * zb + xa * yb - ya * xb + za * wb,
            ],
            dtype=np.float32,
        )
        n = float(np.linalg.norm(out))
        if n > 1e-8:
            out = out / n
        else:
            out = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return out.astype(np.float32)

    def _capture_or_apply_initial_yaw(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        pv = frame.get("Pelvis")
        if not isinstance(pv, (list, tuple)) or len(pv) < 2:
            return frame
        try:
            pelvis_q = np.asarray(pv[1], dtype=np.float32).reshape(4)
        except Exception:
            return frame

        yaw_inv = self._gmr_initial_yaw_inv_wxyz
        if yaw_inv is None:
            # Z-up yaw from scalar-first quaternion [w, x, y, z].
            w, x, y, z = [float(v) for v in pelvis_q]
            yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            half = -0.5 * yaw
            yaw_inv = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)
            n = float(np.linalg.norm(yaw_inv))
            if n > 1e-8:
                yaw_inv = yaw_inv / n
            else:
                yaw_inv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._gmr_initial_yaw_inv_wxyz = yaw_inv.astype(np.float32)

        c = float(yaw_inv[0])
        s = float(yaw_inv[3])
        rot_z = np.array(
            [
                [1.0 - 2.0 * s * s, -2.0 * c * s, 0.0],
                [2.0 * c * s, 1.0 - 2.0 * s * s, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        out: Dict[str, Any] = {}
        for name, v in frame.items():
            if not isinstance(v, (list, tuple)) or len(v) < 2:
                continue
            try:
                p = np.asarray(v[0], dtype=np.float32).reshape(3)
                q = np.asarray(v[1], dtype=np.float32).reshape(4)
            except Exception:
                continue
            out[name] = [(rot_z @ p).astype(np.float32), self._quat_mul_wxyz(yaw_inv, q)]
        return out

    def to_gmr_human_frame(self, body_frame: Any) -> Dict[str, Any]:
        if not isinstance(body_frame, dict):
            return {}
        out: Dict[str, Any] = {}
        for src_name, dst_name in self._XSENS_TO_GMR_NAME.items():
            v = body_frame.get(src_name, None)
            if not isinstance(v, (list, tuple)) or len(v) < 2:
                continue
            try:
                p = np.asarray(v[0], dtype=np.float32).reshape(3)
                q = np.asarray(v[1], dtype=np.float32).reshape(4)
            except Exception:
                continue
            out[dst_name] = [p, q]
        return self._capture_or_apply_initial_yaw(out)

    def _resolve_quat_order(self, seg_id: int, raw4: np.ndarray) -> np.ndarray:
        raw4 = np.asarray(raw4, dtype=np.float32).reshape(4)
        q_wxyz = _safe_quat_wxyz(raw4)
        q_xyzw = _safe_quat_wxyz(np.array([raw4[3], raw4[0], raw4[1], raw4[2]], dtype=np.float32))

        if self._quat_order == "wxyz":
            return q_wxyz
        if self._quat_order == "xyzw":
            return q_xyzw

        # auto: prefer continuity with previous segment quaternion.
        q_prev = self._last_quat_by_seg.get(int(seg_id))
        if q_prev is not None:
            d_wxyz = abs(float(np.dot(q_prev, q_wxyz)))
            d_xyzw = abs(float(np.dot(q_prev, q_xyzw)))
            return q_wxyz if d_wxyz >= d_xyzw else q_xyzw
        # cold start: prefer larger |w| (smaller rotation from identity).
        return q_wxyz if abs(float(q_wxyz[0])) >= abs(float(q_xyzw[0])) else q_xyzw

    def _prune_pending(self) -> None:
        now = time.time()
        stale = [
            sample
            for sample, state in self._pending.items()
            if (now - float(state.updated_ts)) > float(self._STALE_SECONDS)
        ]
        for sample in stale:
            self._pending.pop(sample, None)
        if len(self._pending) > self._MAX_PENDING:
            ordered = sorted(self._pending.items(), key=lambda kv: float(kv[1].updated_ts))
            for sample, _state in ordered[: max(0, len(self._pending) - self._MAX_PENDING)]:
                self._pending.pop(sample, None)

    def _decode_datagram(self, datagram: bytes) -> Dict[str, Any]:
        if len(datagram) < self._HEADER.size:
            return {"ok": False, "reason": "packet_too_short"}
        (
            id_string,
            sample_counter,
            datagram_counter_raw,
            num_items,
            _time_code,
            character_id,
            num_body_segments,
            num_props,
            _num_fingers,
            _reserved,
            payload_size,
        ) = self._HEADER.unpack_from(datagram, 0)

        if id_string[:4] != b"MXTP":
            return {"ok": False, "reason": "invalid_magic"}
        msg_type = id_string[4:6].decode("ascii", errors="ignore")
        if msg_type != "02":
            return {"ok": False, "reason": f"unsupported_type:{msg_type}"}
        if int(self.cfg.character_id_filter) >= 0 and int(character_id) != int(self.cfg.character_id_filter):
            return {"ok": False, "reason": "character_mismatch"}

        payload_begin = self._HEADER.size
        payload_expected_end = payload_begin + int(payload_size)
        payload_end = min(len(datagram), payload_expected_end)
        payload = datagram[payload_begin:payload_end]
        max_items_from_size = len(payload) // self._ITEM.size
        item_count = min(int(num_items), int(max_items_from_size))

        parts_xyz_q: Dict[int, list[np.ndarray]] = {}
        parts_q_xyz: Dict[int, list[np.ndarray]] = {}
        score_xyz_q = 0.0
        score_q_xyz = 0.0
        off = 0
        for _ in range(item_count):
            seg_id, v1, v2, v3, v4, v5, v6, v7 = self._ITEM.unpack_from(payload, off)
            off += self._ITEM.size
            if int(seg_id) not in self._BODY_ID_TO_NAME:
                continue
            p_a, q_a, s_a, p_b, q_b, s_b = _decode_type02_item_candidates(v1, v2, v3, v4, v5, v6, v7)
            score_xyz_q += float(s_a)
            score_q_xyz += float(s_b)
            parts_xyz_q[int(seg_id)] = [p_a, q_a]
            parts_q_xyz[int(seg_id)] = [p_b, q_b]

        if self._type02_layout is None:
            self._type02_layout = "q_xyz" if score_q_xyz < score_xyz_q else "xyz_q"
            print(f"[xsens_adapter] auto type02 layout={self._type02_layout}")

        raw_parts = parts_q_xyz if self._type02_layout == "q_xyz" else parts_xyz_q
        parts: Dict[int, list[np.ndarray]] = {}
        for seg_id, pair in raw_parts.items():
            pos_raw, quat_raw = pair
            quat_wxyz = self._resolve_quat_order(int(seg_id), np.asarray(quat_raw, dtype=np.float32).reshape(4))
            if self._position_scale_to_m is None:
                # Auto-detect whether stream position is already in meters or in centimeters.
                raw_mag = float(np.max(np.abs(pos_raw)))
                self._position_scale_to_m = 0.01 if raw_mag > 10.0 else 1.0
                unit_name = "cm" if self._position_scale_to_m == 0.01 else "m"
                print(f"[xsens_adapter] auto position unit={unit_name}, scale={self._position_scale_to_m}")
            pos_m = np.asarray(pos_raw, dtype=np.float32) * np.float32(self._position_scale_to_m)
            self._last_quat_by_seg[int(seg_id)] = quat_wxyz
            parts[int(seg_id)] = [pos_m, quat_wxyz]

        return {
            "ok": True,
            "sample_counter": int(sample_counter),
            "datagram_index": int(datagram_counter_raw & 0x7F),
            "is_last": bool(datagram_counter_raw & 0x80),
            "num_body_segments": int(num_body_segments),
            "num_props": int(num_props),
            "parts": parts,
        }

    def _consume_datagram(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sample = int(d["sample_counter"])
        if sample <= self._last_frame_index:
            # In offline pcap loop replay, sample counters restart from the beginning.
            # Detect a large backward jump and treat it as a stream restart.
            if (self._last_frame_index - sample) >= int(self._SAMPLE_RESET_THRESHOLD):
                self._pending.clear()
                self._last_frame_index = sample - 1
            else:
                return None

        st = self._pending.get(sample)
        if st is None:
            st = _SampleState(segments={}, updated_ts=time.time())
            self._pending[sample] = st

        st.updated_ts = time.time()
        st.segments.update(d["parts"])

        part_idx = int(d["datagram_index"])
        st.max_part_index = max(st.max_part_index, part_idx)
        if bool(d["is_last"]):
            st.got_last = True

        expected_body = max(1, min(23, int(d.get("num_body_segments", 23))))
        got_body = all(seg_id in st.segments for seg_id in range(1, expected_body + 1))

        if got_body and (st.got_last or not bool(self.cfg.drop_incomplete_samples)):
            body_frame_raw: Dict[str, Any] = {}
            for seg_id in range(1, expected_body + 1):
                name = self._BODY_ID_TO_NAME[seg_id]
                body_frame_raw[name] = st.segments[seg_id]
            self._pending.pop(sample, None)
            self._last_frame_index = sample
            return {"ok": True, "frame_index": sample, "body_frame": body_frame_raw}

        if st.got_last and bool(self.cfg.drop_incomplete_samples):
            # Final packet arrived but sample is incomplete.
            self._pending.pop(sample, None)
        return None

    def read_frame(self) -> Dict[str, Any]:
        if (not self._initialized) or (self.sock is None):
            return {"ok": False, "reason": "not_initialized"}

        self._prune_pending()
        deadline = time.time() + max(0.005, float(self.cfg.socket_timeout_s))
        while time.time() <= deadline:
            try:
                datagram, _addr = self.sock.recvfrom(65535)
            except socket.timeout:
                break
            except Exception as e:
                return {"ok": False, "reason": f"socket_error:{e}"}

            d = self._decode_datagram(datagram)
            if not bool(d.get("ok", False)):
                continue
            out = self._consume_datagram(d)
            if out is not None:
                return out
        return {"ok": False, "reason": "no_update"}

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self._initialized = False
        self._pending = {}
        self._gmr_initial_yaw_inv_wxyz = None
