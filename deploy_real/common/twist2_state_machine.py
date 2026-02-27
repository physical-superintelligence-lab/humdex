from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class Twist2LoopState:
    ramp_active: bool
    ramp_exit_mode: bool
    ramp_t0: float | None
    ramp_seconds: float
    ramp_from_body_35: np.ndarray
    ramp_from_neck_2: np.ndarray
    hold_frozen_body_35: np.ndarray | None
    hold_frozen_neck_2: np.ndarray | None
    safe_idle_seq_np: List[np.ndarray]
    last_send_enabled: bool | None
    last_hold_enabled: bool | None


def init_loop_state(
    *,
    last_pub_body_35: List[float],
    last_pub_neck_2: List[float],
    safe_idle_body_35: List[float],
    safe_idle_seq_35: List[List[float]],
    start_ramp_seconds: float,
    now_s: float,
) -> Twist2LoopState:
    ramp_from_body = np.asarray(last_pub_body_35, dtype=float).reshape(-1)
    ramp_from_neck = np.asarray(last_pub_neck_2, dtype=float).reshape(-1)
    ramp_active = False
    ramp_exit_mode = False
    ramp_t0 = None
    ramp_seconds = 0.0
    if float(start_ramp_seconds) > 1e-6:
        ramp_active = True
        ramp_exit_mode = False
        ramp_t0 = float(now_s)
        ramp_seconds = float(start_ramp_seconds)
        ramp_from_body = np.asarray(safe_idle_body_35, dtype=float).reshape(-1)
        ramp_from_neck = np.asarray([0.0, 0.0], dtype=float).reshape(-1)
    safe_seq = [np.asarray(x, dtype=float).reshape(-1) for x in safe_idle_seq_35]
    return Twist2LoopState(
        ramp_active=ramp_active,
        ramp_exit_mode=ramp_exit_mode,
        ramp_t0=ramp_t0,
        ramp_seconds=ramp_seconds,
        ramp_from_body_35=ramp_from_body,
        ramp_from_neck_2=ramp_from_neck,
        hold_frozen_body_35=None,
        hold_frozen_neck_2=None,
        safe_idle_seq_np=safe_seq,
        last_send_enabled=None,
        last_hold_enabled=None,
    )


def on_keyboard_state(
    *,
    loop: Twist2LoopState,
    send_enabled: bool,
    hold_enabled: bool,
    exit_requested: bool,
    last_pub_body_35: List[float],
    last_pub_neck_2: List[float],
    toggle_ramp_seconds: float,
    exit_ramp_seconds: float,
    now_s: float,
) -> Tuple[bool, bool, bool]:
    if loop.last_send_enabled is None:
        loop.last_send_enabled = bool(send_enabled)
    if loop.last_hold_enabled is None:
        loop.last_hold_enabled = bool(hold_enabled)
    toggled = (bool(send_enabled) != bool(loop.last_send_enabled)) or (
        bool(hold_enabled) != bool(loop.last_hold_enabled)
    )

    if (not bool(loop.last_hold_enabled)) and bool(hold_enabled):
        loop.hold_frozen_body_35 = np.asarray(last_pub_body_35, dtype=float).reshape(-1).copy()
        loop.hold_frozen_neck_2 = np.asarray(last_pub_neck_2, dtype=float).reshape(-1).copy()
    if bool(loop.last_hold_enabled) and (not bool(hold_enabled)):
        loop.hold_frozen_body_35 = None
        loop.hold_frozen_neck_2 = None

    if bool(exit_requested) and float(exit_ramp_seconds) > 1e-6 and (not loop.ramp_exit_mode):
        loop.ramp_active = True
        loop.ramp_exit_mode = True
        loop.ramp_t0 = float(now_s)
        loop.ramp_seconds = float(exit_ramp_seconds)
        loop.ramp_from_body_35 = np.asarray(last_pub_body_35, dtype=float).reshape(-1).copy()
        loop.ramp_from_neck_2 = np.asarray(last_pub_neck_2, dtype=float).reshape(-1).copy()
        send_enabled = False
        hold_enabled = False
    elif toggled and float(toggle_ramp_seconds) > 1e-6:
        loop.ramp_active = True
        loop.ramp_exit_mode = False
        loop.ramp_t0 = float(now_s)
        loop.ramp_seconds = float(toggle_ramp_seconds)
        loop.ramp_from_body_35 = np.asarray(last_pub_body_35, dtype=float).reshape(-1).copy()
        loop.ramp_from_neck_2 = np.asarray(last_pub_neck_2, dtype=float).reshape(-1).copy()

    loop.last_send_enabled = bool(send_enabled)
    loop.last_hold_enabled = bool(hold_enabled)
    return bool(send_enabled), bool(hold_enabled), bool(exit_requested and (not loop.ramp_exit_mode))


def apply_hold_default_and_ramp(
    *,
    loop: Twist2LoopState,
    send_enabled: bool,
    hold_enabled: bool,
    retarget_body_35: List[float],
    retarget_neck_2: List[float],
    ramp_ease: str,
    now_s: float,
    ease_fn: Callable[[float, str], float],
) -> Tuple[List[float], List[float], bool]:
    out_body = np.asarray(retarget_body_35, dtype=float).reshape(-1)
    out_neck = np.asarray(retarget_neck_2, dtype=float).reshape(-1)

    if bool(hold_enabled) and (loop.hold_frozen_body_35 is not None):
        out_body = loop.hold_frozen_body_35.reshape(-1).copy()
        if loop.hold_frozen_neck_2 is not None:
            out_neck = loop.hold_frozen_neck_2.reshape(-1).copy()
        else:
            out_neck = np.asarray([0.0, 0.0], dtype=float).reshape(-1)
    elif not bool(send_enabled):
        out_body = loop.safe_idle_seq_np[-1].reshape(-1).copy()
        out_neck = np.asarray([0.0, 0.0], dtype=float).reshape(-1)

    exit_ramp_done = False
    if loop.ramp_active and (loop.ramp_t0 is not None) and loop.ramp_seconds > 1e-6:
        alpha = (float(now_s) - float(loop.ramp_t0)) / max(1e-6, float(loop.ramp_seconds))
        w = ease_fn(alpha, ease=str(ramp_ease))
        out_body = (1.0 - w) * loop.ramp_from_body_35 + w * out_body
        out_neck = (1.0 - w) * loop.ramp_from_neck_2 + w * out_neck
        if alpha >= 1.0:
            loop.ramp_active = False
            loop.ramp_t0 = None
            loop.ramp_seconds = 0.0
            if loop.ramp_exit_mode:
                exit_ramp_done = True

    return out_body.reshape(-1).tolist(), out_neck.reshape(-1).tolist(), exit_ramp_done
