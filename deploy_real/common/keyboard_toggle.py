from __future__ import annotations

import select
import subprocess
import sys
import termios
import threading
import tty
from typing import Any, Optional, Tuple

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_CYAN = "\033[36m"

class KeyboardToggle:
    def __init__(
        self,
        enable: bool,
        toggle_send_key: str = "k",
        hold_key: str = "p",
        exit_key: str = "q",
        emergency_stop_key: str = "e",
        backend: str = "stdin",
        evdev_device: str = "auto",
        evdev_grab: bool = False,
    ) -> None:
        self.enable = bool(enable)
        self.toggle_send_key = (toggle_send_key or "k")[0]
        self.hold_key = (hold_key or "p")[0]
        self.exit_key = (exit_key or "q")[0]
        self.emergency_stop_key = (emergency_stop_key or "e")[0]
        self.backend = (backend or "stdin").strip().lower()
        self.evdev_device = str(evdev_device).strip() if evdev_device is not None else "auto"
        self.evdev_grab = bool(evdev_grab)

        self._send_enabled = True
        self._hold_enabled = False
        self._exit_requested = False
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stdin_fd: Optional[int] = None
        self._stdin_old: Any = None

    def _print_status(self, *, trigger_key: str) -> None:
        send_enabled, hold_enabled, exit_requested = self.get_extended_state()
        if exit_requested:
            banner_style = f"{_ANSI_BOLD}{_ANSI_RED}"
        elif hold_enabled:
            banner_style = f"{_ANSI_BOLD}{_ANSI_YELLOW}"
        elif send_enabled:
            banner_style = f"{_ANSI_BOLD}{_ANSI_GREEN}"
        else:
            banner_style = f"{_ANSI_BOLD}{_ANSI_RED}"
        send_txt = "ON" if send_enabled else "OFF"
        hold_txt = "ON" if hold_enabled else "OFF"
        exit_txt = "YES" if exit_requested else "NO"
        msg = (
            f"[KEYBOARD] KEY={trigger_key} "
            f"SEND={send_txt} HOLD={hold_txt} EXIT={exit_txt}"
        )
        line = "=" * max(24, len(msg))
        print(
            f"\n{banner_style}{line}{_ANSI_RESET}\n"
            f"{banner_style}{msg}{_ANSI_RESET}\n"
            f"{banner_style}{line}{_ANSI_RESET}",
            flush=True,
        )

    def start(self) -> None:
        if not self.enable:
            return
        if self.backend in ["evdev", "both"]:
            # Keep teleop independent from deploy_real/data_utils/evdev_hotkeys.py.
            # If evdev is requested, degrade to stdin so startup never fails.
            print("[keyboard_toggle] evdev backend is disabled; fallback to stdin")
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass
        self._thread = None
        try:
            if self._stdin_fd is not None and self._stdin_old is not None:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old)
        except Exception:
            pass

    def get_extended_state(self) -> Tuple[bool, bool, bool]:
        with self._lock:
            return (
                bool(self._send_enabled),
                bool(self._hold_enabled),
                bool(self._exit_requested),
            )

    def _emergency_stop(self) -> None:
        try:
            subprocess.run(["pkill", "-f", "sim2real.sh"], capture_output=True, text=True, timeout=5)
            subprocess.run(["pkill", "-f", "server_low_level_g1_real_future.py"], capture_output=True, text=True, timeout=5)
        except Exception:
            pass

    def _handle_key(self, ch: str) -> None:
        ch = (ch or "")[0]
        if ch == self.toggle_send_key:
            with self._lock:
                self._send_enabled = not self._send_enabled
                if not self._send_enabled:
                    self._hold_enabled = False
            self._print_status(trigger_key=ch)
        elif ch == self.hold_key:
            with self._lock:
                if not self._send_enabled:
                    self._hold_enabled = False
                else:
                    self._hold_enabled = not self._hold_enabled
            self._print_status(trigger_key=ch)
        elif ch == self.exit_key:
            with self._lock:
                self._exit_requested = True
            self._print_status(trigger_key=ch)
        elif ch == self.emergency_stop_key:
            self._emergency_stop()
            msg = "[KEYBOARD] EMERGENCY STOP REQUESTED"
            line = "=" * len(msg)
            style = f"{_ANSI_BOLD}{_ANSI_RED}"
            print(f"\n{style}{line}{_ANSI_RESET}\n{style}{msg}{_ANSI_RESET}\n{style}{line}{_ANSI_RESET}", flush=True)

    def _loop(self) -> None:
        try:
            fd = sys.stdin.fileno()
            self._stdin_fd = fd
            self._stdin_old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                self._handle_key(ch)
        finally:
            try:
                if self._stdin_fd is not None and self._stdin_old is not None:
                    termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old)
            except Exception:
                pass
