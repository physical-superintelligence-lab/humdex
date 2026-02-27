"""
全局键盘热键（Linux）：通过 /dev/input/event* 读取按键事件，不依赖窗口/终端前台焦点。

注意：
- 仅在 Linux 上工作。
- 需要对 /dev/input/event* 有读权限（通常需要 root 或把用户加入 input 组）。
- Wayland/X11 无关，因为直接读 evdev 设备。
"""

from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EvdevHotkeyConfig:
    device: str = "auto"  # "/dev/input/eventX" 或 "auto"
    grab: bool = False    # 是否 grab（避免按键被其他程序同时收到；谨慎使用）


class EvdevHotkeys:
    """
    后台线程监听键盘按键（key down），触发回调：callback(ch: str)

    - ch 为小写单字符，比如 'r','k','p','q'。
    """

    def __init__(self, cfg: EvdevHotkeyConfig, callback: Callable[[str], None]) -> None:
        self.cfg = cfg
        self.callback = callback
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._device_path: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        # 在主线程里先解析/验证设备，避免后台线程异常“悄悄死掉”
        path = self._resolve_device_path()
        self._device_path = path
        try:
            from evdev import InputDevice  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("未安装 python-evdev。请安装：pip install evdev（或在 conda env 里安装）。") from e
        try:
            _ = InputDevice(path)
        except PermissionError as e:
            raise PermissionError(
                f"无权限读取 {path}。请用 sudo 运行，或把用户加入 input 组（然后重新登录）。"
            ) from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f"找不到设备：{path}") from e
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

    def _resolve_device_path(self) -> str:
        dev = str(self.cfg.device).strip()
        if dev and dev != "auto":
            return dev
        try:
            from evdev import InputDevice, list_devices  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "未安装 python-evdev。请安装：pip install evdev（或在 conda env 里安装）。"
            ) from e

        # 1) 优先：/dev/input/by-id/*event-kbd（更稳定）
        by_id = [p for p in sorted(glob.glob("/dev/input/by-id/*event-kbd")) if os.path.exists(p)]
        if by_id:
            # 尽量不要默认选脚踏板/特殊输入设备（除非只有它）
            prefer = [p for p in by_id if "footswitch" not in os.path.basename(p).lower()]
            return (prefer[0] if prefer else by_id[0])

        # 2) 再尝试：evdev 自带 list_devices
        try:
            devs = list_devices()
        except Exception:
            devs = []

        # 3) 如果 list_devices 意外为空，自己用 glob 扫 /dev/input/event*
        if not devs:
            devs = sorted(glob.glob("/dev/input/event*"))

        if not devs:
            raise FileNotFoundError("找不到任何 /dev/input/event* 设备（可能在容器内未映射 /dev/input）")

        # 自动找一个“看起来像键盘”的设备
        for path in devs:
            try:
                d = InputDevice(path)
                name = (d.name or "").lower()
                if "keyboard" in name or "kbd" in name:
                    return path
            except Exception:
                continue
        # fallback: 第一个设备（可能不对，所以建议显式指定 --evdev_device）
        return devs[0]

    def _loop(self) -> None:
        try:
            from evdev import InputDevice, categorize, ecodes  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "未安装 python-evdev。请安装：pip install evdev（或在 conda env 里安装）。"
            ) from e

        path = self._device_path or self._resolve_device_path()
        dev = InputDevice(path)
        if bool(self.cfg.grab):
            try:
                dev.grab()
            except Exception:
                # grab 失败不致命
                pass

        # keycode 映射：KEY_R -> 'r'
        def keycode_to_char(code: str) -> Optional[str]:
            c = str(code)
            if c.startswith("KEY_") and len(c) == 5:  # KEY_A .. KEY_Z
                return c[-1].lower()
            if c.startswith("KEY_") and len(c) == 6:  # KEY_0 .. KEY_9
                return c[-1]
            return None

        for event in dev.read_loop():
            if self._stop.is_set():
                break
            if event.type != ecodes.EV_KEY:
                continue
            try:
                ke = categorize(event)
                # key_down=1
                if int(getattr(ke, "keystate", 0)) != 1:
                    continue
                kc = getattr(ke, "keycode", None)
                # keycode 可能是 list（比如 shift+key）
                if isinstance(kc, (list, tuple)):
                    for one in kc:
                        ch = keycode_to_char(str(one))
                        if ch is not None:
                            self.callback(ch)
                            break
                else:
                    ch = keycode_to_char(str(kc))
                    if ch is not None:
                        self.callback(ch)
            except Exception:
                continue


