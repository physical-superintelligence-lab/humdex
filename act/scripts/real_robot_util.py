"""Utilities for real robot deployment: vision, keyboard safety, and toggle ramp."""

import sys
import time
import threading
import select
import termios
import tty
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Vision Client (ZMQ)
# =============================================================================

class VisionReader:
    """Read images from ZMQ stream."""

    def __init__(self, server_ip: str = "192.168.123.164", port: int = 5555,
                 img_shape: tuple = (480, 640, 3),
                 source_bgr: bool = True):
        from multiprocessing import shared_memory
        from data_utils.vision_client import VisionClient

        self.img_shape = img_shape
        self.source_bgr = bool(source_bgr)
        shm_bytes = int(np.prod(img_shape) * np.uint8().itemsize)
        self.shm = shared_memory.SharedMemory(create=True, size=shm_bytes)
        self.image_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)

        self.client = VisionClient(
            server_address=server_ip,
            port=port,
            img_shape=img_shape,
            img_shm_name=self.shm.name,
            image_show=False,
            depth_show=False,
            unit_test=True,
        )

        self.thread = threading.Thread(target=self.client.receive_process, daemon=True)
        self.thread.start()
        print(f"Vision client started: {server_ip}:{port}")

    def get_image(self) -> np.ndarray:
        """Get latest RGB image (H, W, 3)."""
        img = self.image_array.copy()
        if self.source_bgr and img.ndim == 3 and img.shape[2] == 3:
            return img[:, :, ::-1].copy()
        return img

    def close(self):
        try:
            self.client.running = False
            self.thread.join(timeout=1.0)
            self.shm.unlink()
            self.shm.close()
        except:
            pass


# =============================================================================
# Keyboard Toggle (k/p safety semantics)
# =============================================================================

class KeyboardToggle:
    """Terminal keyboard toggles for robot safety control.

    - k: toggle send_enabled. When disabled: publish safe idle action (and disable hold).
    - p: toggle hold_position. When enabled: keep publishing cached last action.
    """

    def __init__(self, enabled: bool, toggle_send_key: str = "k", hold_position_key: str = "p"):
        self.enabled = bool(enabled)
        self.toggle_send_key = (toggle_send_key or "k")[0]
        self.hold_position_key = (hold_position_key or "p")[0]

        self._send_enabled = True
        self._hold_position_enabled = False
        self._lock = threading.Lock()

        self._thread = None
        self._stop = threading.Event()
        self._stdin_fd = None
        self._stdin_old_termios = None

    def start(self):
        if not self.enabled:
            return
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass
        self._thread = None

    def get(self):
        with self._lock:
            return self._send_enabled, self._hold_position_enabled

    def _loop(self):
        try:
            fd = sys.stdin.fileno()
            self._stdin_fd = fd
            self._stdin_old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            print(f"[Keyboard] Press '{self.toggle_send_key}' to toggle send_enabled (disabled => safe idle)")
            print(f"[Keyboard] Press '{self.hold_position_key}' to toggle hold_position (hold last action)")
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == self.toggle_send_key:
                    with self._lock:
                        self._send_enabled = not self._send_enabled
                        enabled = self._send_enabled
                        if not self._send_enabled:
                            self._hold_position_enabled = False
                    print(f"[Keyboard] send_enabled => {enabled}")
                elif ch == self.hold_position_key:
                    with self._lock:
                        if not self._send_enabled:
                            self._hold_position_enabled = False
                            enabled = self._hold_position_enabled
                        else:
                            self._hold_position_enabled = not self._hold_position_enabled
                            enabled = self._hold_position_enabled
                    print(f"[Keyboard] hold_position_enabled => {enabled}")
        except Exception as e:
            print(f"[Keyboard] Keyboard listener unavailable: {e}")
        finally:
            try:
                if self._stdin_fd is not None and self._stdin_old_termios is not None:
                    termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_termios)
            except Exception:
                pass


# =============================================================================
# Safety helpers
# =============================================================================

def get_safe_idle_body_35(robot_key: str) -> np.ndarray:
    """Safe idle 35D action for the given robot_key."""
    from data_utils.params import DEFAULT_MIMIC_OBS
    base = DEFAULT_MIMIC_OBS.get(robot_key, DEFAULT_MIMIC_OBS["unitree_g1"])
    return np.array(base[:35], dtype=np.float32)


def publish_with_kp_safety(
    redis_io,
    kb: KeyboardToggle,
    safe_idle_body_35: np.ndarray,
    cached_body: np.ndarray,
    cached_neck: np.ndarray,
    desired_body: np.ndarray,
    desired_neck: Optional[np.ndarray] = None,
):
    """
    Publish action with k/p safety semantics.

    - k (send_enabled=False): publish safe idle, set wuji mode=default
    - p (hold_enabled=True): publish cached action, set wuji mode=hold
    - else: publish desired action, update cache, set wuji mode=follow

    Returns:
        pub_body, pub_neck, cached_body, cached_neck, advance_step(bool)
    """
    send_enabled, hold_enabled = kb.get()

    if not send_enabled:
        pub_body = safe_idle_body_35
        pub_neck = cached_neck
        redis_io.set_wuji_hand_mode("default")
        advance = False
    elif hold_enabled:
        pub_body = cached_body
        pub_neck = cached_neck
        redis_io.set_wuji_hand_mode("hold")
        advance = False
    else:
        pub_body = np.asarray(desired_body, dtype=np.float32)
        pub_neck = np.asarray(desired_neck if desired_neck is not None else cached_neck, dtype=np.float32)
        cached_body = pub_body.copy()
        cached_neck = pub_neck.copy()
        redis_io.set_wuji_hand_mode("follow")
        advance = True

    redis_io.publish_action(pub_body, pub_neck)
    return pub_body, pub_neck, cached_body, cached_neck, advance


# =============================================================================
# Toggle ramp (smooth transitions)
# =============================================================================

def _ease(alpha: float, ease: str = "cosine") -> float:
    """Easing function for smooth ramp transitions."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if str(ease) == "linear":
        return a
    return 0.5 - 0.5 * float(np.cos(np.pi * a))


@dataclass
class ToggleRamp:
    """Smooth transition on k/p toggles (same semantics as init_pose ramp)."""
    seconds: float = 0.0
    ease: str = "cosine"
    active: bool = False
    t0: float = 0.0
    from_body: np.ndarray = field(default_factory=lambda: np.zeros(35, dtype=np.float32))
    from_neck: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    to_body: np.ndarray = field(default_factory=lambda: np.zeros(35, dtype=np.float32))
    to_neck: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    target_mode: str = "follow"

    def start(
        self,
        from_body: np.ndarray,
        from_neck: np.ndarray,
        to_body: np.ndarray,
        to_neck: np.ndarray,
        target_mode: str,
        seconds: float,
        ease: str,
    ):
        self.seconds = float(seconds)
        self.ease = str(ease)
        self.active = self.seconds > 1e-6
        self.t0 = time.time()
        self.from_body = np.asarray(from_body, dtype=np.float32).reshape(35).copy()
        self.from_neck = np.asarray(from_neck, dtype=np.float32).reshape(2).copy()
        self.to_body = np.asarray(to_body, dtype=np.float32).reshape(35).copy()
        self.to_neck = np.asarray(to_neck, dtype=np.float32).reshape(2).copy()
        self.target_mode = str(target_mode)

    def value(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        if not self.active:
            return self.to_body, self.to_neck, False
        alpha = (time.time() - self.t0) / max(1e-6, float(self.seconds))
        w = _ease(alpha, self.ease)
        body = self.from_body + w * (self.to_body - self.from_body)
        neck = self.from_neck + w * (self.to_neck - self.from_neck)
        done = float(alpha) >= 1.0 - 1e-6
        if done:
            self.active = False
        return body.astype(np.float32), neck.astype(np.float32), (not self.active)
