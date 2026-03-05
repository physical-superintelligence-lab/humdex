from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ManusConfig:
    address: str
    left_glove_sn: Optional[str]
    right_glove_sn: Optional[str]
    auto_assign: bool
    recv_timeout_ms: int
    flip_x: bool
    hands: str = "both"


@dataclass(frozen=True)
class ManusRuntimeConfig:
    pass


@dataclass
class ManusFrame:
    ts: float
    sn: str
    xyz: np.ndarray  # (25,3)
    quat_xyzw: np.ndarray  # (25,4)


class ManusSkeletonReceiver:
    def __init__(
        self,
        *,
        address: str,
        left_glove_sn: Optional[str],
        right_glove_sn: Optional[str],
        auto_assign: bool,
        recv_timeout_ms: int,
        flip_x: bool,
    ) -> None:
        import zmq  # type: ignore

        self.address = str(address)
        self.left_glove_sn = None if left_glove_sn in [None, ""] else str(left_glove_sn)
        self.right_glove_sn = None if right_glove_sn in [None, ""] else str(right_glove_sn)
        self.auto_assign = bool(auto_assign)
        self._x_sign = -1.0 if bool(flip_x) else 1.0

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, int(max(1, recv_timeout_ms)))
        self.socket.connect(self.address)

        self._lock = threading.Lock()
        self._running = True
        self._left: Optional[ManusFrame] = None
        self._right: Optional[ManusFrame] = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="manus-recv")
        self._thread.start()

    @staticmethod
    def _parse_skeleton_176(parts_176: list[str]) -> Tuple[str, np.ndarray, np.ndarray]:
        if len(parts_176) != 176:
            raise ValueError(f"expected 176 fields, got {len(parts_176)}")
        sn = str(parts_176[0])
        floats = np.asarray(list(map(float, parts_176[1:])), dtype=np.float32)
        arr = floats.reshape(25, 7)
        return sn, arr[:, 0:3], arr[:, 3:7]

    def _maybe_auto_assign(self, sn: str) -> None:
        if not self.auto_assign:
            return
        if self.left_glove_sn is None and sn != self.right_glove_sn:
            self.left_glove_sn = sn
            return
        if self.right_glove_sn is None and sn != self.left_glove_sn:
            self.right_glove_sn = sn

    def _apply_flip_x(self, xyz: np.ndarray) -> np.ndarray:
        if self._x_sign == 1.0:
            return xyz
        out = np.asarray(xyz, dtype=np.float32).copy()
        out[:, 0] *= self._x_sign
        return out

    def _update_by_sn(self, frame: ManusFrame) -> None:
        self._maybe_auto_assign(frame.sn)
        if self.left_glove_sn is not None and frame.sn == self.left_glove_sn:
            self._left = frame
        if self.right_glove_sn is not None and frame.sn == self.right_glove_sn:
            self._right = frame

    def _loop(self) -> None:
        import zmq  # type: ignore

        while self._running:
            try:
                msg = self.socket.recv()
            except zmq.Again:
                continue
            except Exception:
                break
            try:
                text = msg.decode("utf-8", errors="replace")
                parts = text.split(",")
                now = time.time()
                with self._lock:
                    if len(parts) == 176:
                        sn, xyz, quat = self._parse_skeleton_176(parts)
                        self._update_by_sn(ManusFrame(ts=now, sn=sn, xyz=self._apply_flip_x(xyz), quat_xyzw=quat))
                    elif len(parts) == 352:
                        sn0, xyz0, quat0 = self._parse_skeleton_176(parts[0:176])
                        sn1, xyz1, quat1 = self._parse_skeleton_176(parts[176:352])
                        self._update_by_sn(ManusFrame(ts=now, sn=sn0, xyz=self._apply_flip_x(xyz0), quat_xyzw=quat0))
                        self._update_by_sn(ManusFrame(ts=now, sn=sn1, xyz=self._apply_flip_x(xyz1), quat_xyzw=quat1))
            except Exception:
                continue

    def get_latest(self) -> Tuple[Optional[ManusFrame], Optional[ManusFrame]]:
        with self._lock:
            return self._left, self._right

    def stop(self) -> None:
        self._running = False
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=0.3)
        except Exception:
            pass
        try:
            self.socket.close(0)
        except Exception:
            pass
        try:
            self.ctx.term()
        except Exception:
            pass


class ManusReader:
    def __init__(self, cfg: ManusConfig) -> None:
        self.cfg = cfg
        self.recv: Optional[ManusSkeletonReceiver] = None
        self._initialized = False
        self._frame_index = -1
        self._last_emitted_ts = 0.0

    def initialize(self) -> None:
        self.recv = ManusSkeletonReceiver(
            address=str(self.cfg.address),
            left_glove_sn=self.cfg.left_glove_sn,
            right_glove_sn=self.cfg.right_glove_sn,
            auto_assign=bool(self.cfg.auto_assign),
            recv_timeout_ms=int(self.cfg.recv_timeout_ms),
            flip_x=bool(self.cfg.flip_x),
        )
        self._initialized = True

    def read_frame(self) -> Dict[str, Any]:
        if (not self._initialized) or (self.recv is None):
            return {"ok": False, "reason": "not_initialized"}
        left, right = self.recv.get_latest()
        hands = str(self.cfg.hands).lower()
        chosen_ts = 0.0
        frame: Dict[str, Any] = {}
        if hands in ["left", "both"] and left is not None:
            frame["manus_left_xyz25"] = np.asarray(left.xyz, dtype=np.float32).reshape(25, 3)
            chosen_ts = max(chosen_ts, float(left.ts))
        if hands in ["right", "both"] and right is not None:
            frame["manus_right_xyz25"] = np.asarray(right.xyz, dtype=np.float32).reshape(25, 3)
            chosen_ts = max(chosen_ts, float(right.ts))
        if len(frame) == 0:
            return {"ok": False, "reason": "no_update"}
        if chosen_ts <= self._last_emitted_ts:
            return {"ok": False, "reason": "no_update"}
        self._last_emitted_ts = chosen_ts
        self._frame_index += 1
        return {"ok": True, "frame_index": int(self._frame_index), "hand_frame": frame}

    def close(self) -> None:
        try:
            if self.recv is not None:
                self.recv.stop()
        except Exception:
            pass
        self.recv = None
        self._initialized = False


def _make_tracking_joint(pos: np.ndarray) -> list:
    p = np.asarray(pos, dtype=np.float32).reshape(3)
    return [p.reshape(-1).tolist(), [1.0, 0.0, 0.0, 0.0]]


def _manus25_to_tracking26(xyz25: np.ndarray, side: str) -> Dict[str, list]:
    side = str(side).lower()
    assert side in ["left", "right"]
    pfx = "Left" if side == "left" else "Right"
    pts = np.asarray(xyz25, dtype=np.float32).reshape(25, 3)
    out: Dict[str, list] = {}
    wrist = pts[0]
    out[f"{pfx}HandWrist"] = _make_tracking_joint(wrist)
    out[f"{pfx}HandPalm"] = _make_tracking_joint(wrist)
    out[f"{pfx}HandThumbMetacarpal"] = _make_tracking_joint(pts[21])
    out[f"{pfx}HandThumbProximal"] = _make_tracking_joint(pts[22])
    out[f"{pfx}HandThumbDistal"] = _make_tracking_joint(pts[23])
    out[f"{pfx}HandThumbTip"] = _make_tracking_joint(pts[24])
    out[f"{pfx}HandIndexMetacarpal"] = _make_tracking_joint(pts[1])
    out[f"{pfx}HandIndexProximal"] = _make_tracking_joint(pts[2])
    out[f"{pfx}HandIndexIntermediate"] = _make_tracking_joint(pts[3])
    out[f"{pfx}HandIndexDistal"] = _make_tracking_joint(pts[4])
    out[f"{pfx}HandIndexTip"] = _make_tracking_joint(pts[5])
    out[f"{pfx}HandMiddleMetacarpal"] = _make_tracking_joint(pts[6])
    out[f"{pfx}HandMiddleProximal"] = _make_tracking_joint(pts[7])
    out[f"{pfx}HandMiddleIntermediate"] = _make_tracking_joint(pts[8])
    out[f"{pfx}HandMiddleDistal"] = _make_tracking_joint(pts[9])
    out[f"{pfx}HandMiddleTip"] = _make_tracking_joint(pts[10])
    out[f"{pfx}HandRingMetacarpal"] = _make_tracking_joint(pts[16])
    out[f"{pfx}HandRingProximal"] = _make_tracking_joint(pts[17])
    out[f"{pfx}HandRingIntermediate"] = _make_tracking_joint(pts[18])
    out[f"{pfx}HandRingDistal"] = _make_tracking_joint(pts[19])
    out[f"{pfx}HandRingTip"] = _make_tracking_joint(pts[20])
    out[f"{pfx}HandLittleMetacarpal"] = _make_tracking_joint(pts[11])
    out[f"{pfx}HandLittleProximal"] = _make_tracking_joint(pts[12])
    out[f"{pfx}HandLittleIntermediate"] = _make_tracking_joint(pts[13])
    out[f"{pfx}HandLittleDistal"] = _make_tracking_joint(pts[14])
    out[f"{pfx}HandLittleTip"] = _make_tracking_joint(pts[15])
    return out


def build_manus_tracking(
    *,
    hands_mode: str,
    is_active: bool,
    now_ms: int,
    fr_hand: Optional[Dict[str, Any]],
    cfg_hand_fk: bool,
    runtime: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    _ = cfg_hand_fk
    _ = runtime
    ht_l: Dict[str, Any] = {"is_active": bool(is_active and hands_mode in ["left", "both"]), "timestamp": now_ms}
    ht_r: Dict[str, Any] = {"is_active": bool(is_active and hands_mode in ["right", "both"]), "timestamp": now_ms}
    if (not isinstance(fr_hand, dict)) or (not bool(is_active)):
        return ht_l, ht_r
    if hands_mode in ["left", "both"] and "manus_left_xyz25" in fr_hand:
        ht_l.update(_manus25_to_tracking26(np.asarray(fr_hand["manus_left_xyz25"], dtype=np.float32), "left"))
    if hands_mode in ["right", "both"] and "manus_right_xyz25" in fr_hand:
        ht_r.update(_manus25_to_tracking26(np.asarray(fr_hand["manus_right_xyz25"], dtype=np.float32), "right"))
    return ht_l, ht_r


def build_manus_runtime(_cfg: ManusRuntimeConfig) -> Dict[str, Any]:
    return {
        "build_hand_tracking": build_manus_tracking,
    }

