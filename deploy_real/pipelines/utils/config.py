from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass
class CommonRuntimeConfig:
    dst_ip: str
    dst_port: int
    local_port: int
    mocap_index: int
    world_space: int
    dst_ip_body: str
    dst_port_body: int
    mocap_index_body: int
    dst_ip_hand: str
    dst_port_hand: int
    mocap_index_hand: int
    redis_ip: str
    actual_human_height: float
    target_fps: float
    hands: str
    format: str
    offset_to_ground: bool
    toggle_send_key: str
    hold_position_key: str
    safe_idle_pose_id: str
    ramp_ease: str
    keyboard_toggle_send: bool
    keyboard_backend: str
    hand_source_timeout_s: float
    config_yaml: str
    hand_fk: bool
    hand_fk_end_site_scale: str
    smooth: bool
    smooth_window_size: int
    print_every: int
    toggle_ramp_seconds: float
    exit_ramp_seconds: float
    start_ramp_seconds: float
    evdev_device: str
    evdev_grab: bool
    vmc_ip: str
    vmc_port: int
    vmc_timeout_s: float
    vmc_rot_mode: str
    vmc_use_fk: bool
    vmc_use_viewer_fk: bool
    vmc_fk_skeleton: str
    vmc_bvh_path: str
    vmc_bvh_scale: float
    manus_address: str
    manus_left_sn: str
    manus_right_sn: str
    manus_auto_assign: bool
    manus_recv_timeout_ms: int
    manus_flip_x: bool
    max_steps: int


@dataclass
class Twist2RuntimeConfig(CommonRuntimeConfig):
    pass


@dataclass
class SonicRuntimeConfig(CommonRuntimeConfig):
    zmq_bind_host: str
    zmq_pose_port: int
    zmq_pose_topic: str
    zmq_num_frames_to_send: int
    zmq_frame_index_step: int
    zmq_use_mailbox: bool
    zmq_catch_up: bool


_MISSING = object()
_ALLOWED_TOP_LEVEL = {"runtime", "network", "retarget", "control", "adapters", "policy"}
_LEGACY_FLAT_KEYS = {
    "redis_ip",
    "dst_ip",
    "dst_port",
    "local_port",
    "mocap_index",
    "world_space",
    "dst_ip_body",
    "dst_port_body",
    "mocap_index_body",
    "dst_ip_hand",
    "dst_port_hand",
    "mocap_index_hand",
    "actual_human_height",
    "target_fps",
    "hands",
    "format",
    "offset_to_ground",
    "toggle_send_key",
    "hold_position_key",
    "safe_idle_pose_id",
    "ramp_ease",
    "keyboard_toggle_send",
    "keyboard_backend",
    "hand_source_timeout_s",
    "hand_fk",
    "hand_fk_end_site_scale",
    "smooth",
    "smooth_window_size",
    "print_every",
    "toggle_ramp_seconds",
    "exit_ramp_seconds",
    "start_ramp_seconds",
    "evdev_device",
    "evdev_grab",
    "vmc_ip",
    "vmc_port",
    "vmc_timeout_s",
    "vmc_rot_mode",
    "vmc_use_fk",
    "vmc_use_viewer_fk",
    "vmc_fk_skeleton",
    "vmc_bvh_path",
    "vmc_bvh_scale",
    "manus_address",
    "manus_left_sn",
    "manus_right_sn",
    "manus_auto_assign",
    "manus_recv_timeout_ms",
    "manus_flip_x",
    "zmq_bind_host",
    "zmq_pose_port",
    "zmq_pose_topic",
    "zmq_num_frames_to_send",
    "zmq_frame_index_step",
    "zmq_use_mailbox",
    "zmq_catch_up",
    "max_steps",
}
_COMMON_REQUIRED_PATHS = [
    "runtime.target_fps",
    "runtime.print_every",
    "runtime.max_steps",
    "network.redis.ip",
    "network.mocap.default.ip",
    "network.mocap.default.port",
    "network.mocap.default.index",
    "network.mocap.default.world_space",
    "network.mocap.default.local_port",
    "network.mocap.body.ip",
    "network.mocap.body.port",
    "network.mocap.body.index",
    "network.mocap.hand.ip",
    "network.mocap.hand.port",
    "network.mocap.hand.index",
    "retarget.actual_human_height",
    "retarget.hands",
    "retarget.format",
    "retarget.offset_to_ground",
    "control.safe_idle_pose_id",
    "control.ramp_ease",
    "control.start_ramp_seconds",
    "control.toggle_ramp_seconds",
    "control.exit_ramp_seconds",
    "policy.twist2.smooth",
    "policy.twist2.smooth_window_size",
    "control.keyboard_toggle_send",
    "control.keyboard_backend",
    "control.toggle_send_key",
    "control.hold_position_key",
    "control.hand_source_timeout_s",
    "control.evdev_device",
    "control.evdev_grab",
    "adapters.vdhand.hand_fk",
    "adapters.vdhand.hand_fk_end_site_scale",
    "adapters.manus.address",
    "adapters.manus.left_sn",
    "adapters.manus.right_sn",
    "adapters.manus.auto_assign",
    "adapters.manus.recv_timeout_ms",
    "adapters.manus.flip_x",
    "adapters.slimevr.vmc_ip",
    "adapters.slimevr.vmc_port",
    "adapters.slimevr.vmc_timeout_s",
    "adapters.slimevr.vmc_rot_mode",
    "adapters.slimevr.vmc_use_fk",
    "adapters.slimevr.vmc_use_viewer_fk",
    "adapters.slimevr.vmc_fk_skeleton",
    "adapters.slimevr.vmc_bvh_path",
    "adapters.slimevr.vmc_bvh_scale",
]
_SONIC_REQUIRED_PATHS = [
    "policy.sonic.zmq_bind_host",
    "policy.sonic.zmq_pose_port",
    "policy.sonic.zmq_pose_topic",
    "policy.sonic.zmq_num_frames_to_send",
    "policy.sonic.zmq_frame_index_step",
    "policy.sonic.zmq_use_mailbox",
    "policy.sonic.zmq_catch_up",
]


def _load_yaml(passthrough: list[str], repo_root: Path, default_config: str) -> tuple[Path, Dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config_yaml", type=str, default=default_config)
    known, _unknown = parser.parse_known_args(passthrough)
    yaml_path = Path(str(known.config_yaml)).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = (repo_root / yaml_path).resolve()
    yaml_cfg: Dict[str, Any] = {}
    if yaml_path.exists():
        import yaml  # type: ignore

        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                yaml_cfg = raw
    return yaml_path, yaml_cfg


def _get_nested(cfg: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    current: Any = cfg
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            if default is _MISSING:
                raise ValueError(f"Missing required config path: {path}")
            return default
        current = current[key]
    return current


def _resolve_repo_path(repo_root: Path, value: str) -> str:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())


def _validate_nested_schema(yaml_cfg: Dict[str, Any], yaml_path: Path) -> None:
    if len(yaml_cfg) == 0:
        raise ValueError(f"Config yaml is empty or invalid: {yaml_path}")

    legacy = sorted(_LEGACY_FLAT_KEYS.intersection(set(yaml_cfg.keys())))
    if legacy:
        preview = ", ".join(legacy[:8])
        raise ValueError(
            "Legacy flat config keys are no longer supported. "
            f"Found at root: {preview}. "
            "Use nested sections: runtime/network/retarget/control/adapters/policy."
        )

    unknown = sorted(set(yaml_cfg.keys()) - _ALLOWED_TOP_LEVEL)
    if unknown:
        raise ValueError(
            "Unknown top-level config sections: "
            f"{', '.join(unknown)}. "
            f"Allowed sections: {', '.join(sorted(_ALLOWED_TOP_LEVEL))}."
        )

    for section in sorted(_ALLOWED_TOP_LEVEL):
        section_val = yaml_cfg.get(section)
        if not isinstance(section_val, dict):
            raise ValueError(f"Config section '{section}' must be a mapping/dict")

    for section_path in [
        "network.redis",
        "network.mocap",
        "network.mocap.default",
        "network.mocap.body",
        "network.mocap.hand",
        "adapters.vdmocap",
        "adapters.vdhand",
        "adapters.manus",
        "adapters.slimevr",
    ]:
        val = _get_nested(yaml_cfg, section_path)
        if not isinstance(val, Mapping):
            raise ValueError(f"Config section '{section_path}' must be a mapping/dict")


def _assert_required_paths(yaml_cfg: Dict[str, Any], paths: list[str]) -> None:
    missing: list[str] = []
    for p in paths:
        try:
            _get_nested(yaml_cfg, p)
        except ValueError:
            missing.append(p)
    if missing:
        sample = ", ".join(missing[:8])
        raise ValueError(f"Missing required config paths: {sample}")


def _build_common_values(yaml_cfg: Dict[str, Any], yaml_path: Path, *, repo_root: Path) -> Dict[str, Any]:
    g = lambda p: _get_nested(yaml_cfg, p)
    return {
        "config_yaml": str(yaml_path),
        "redis_ip": str(g("network.redis.ip")),
        "dst_ip": str(g("network.mocap.default.ip")),
        "dst_port": int(g("network.mocap.default.port")),
        "local_port": int(g("network.mocap.default.local_port")),
        "mocap_index": int(g("network.mocap.default.index")),
        "world_space": int(g("network.mocap.default.world_space")),
        "dst_ip_body": str(g("network.mocap.body.ip")),
        "dst_port_body": int(g("network.mocap.body.port")),
        "mocap_index_body": int(g("network.mocap.body.index")),
        "dst_ip_hand": str(g("network.mocap.hand.ip")),
        "dst_port_hand": int(g("network.mocap.hand.port")),
        "mocap_index_hand": int(g("network.mocap.hand.index")),
        "actual_human_height": float(g("retarget.actual_human_height")),
        "target_fps": float(g("runtime.target_fps")),
        "hands": str(g("retarget.hands")),
        "format": str(g("retarget.format")),
        "offset_to_ground": bool(g("retarget.offset_to_ground")),
        "toggle_send_key": str(g("control.toggle_send_key")),
        "hold_position_key": str(g("control.hold_position_key")),
        "safe_idle_pose_id": str(g("control.safe_idle_pose_id")),
        "ramp_ease": str(g("control.ramp_ease")),
        "keyboard_toggle_send": bool(g("control.keyboard_toggle_send")),
        "keyboard_backend": str(g("control.keyboard_backend")).strip().lower(),
        "hand_source_timeout_s": float(g("control.hand_source_timeout_s")),
        "hand_fk": bool(g("adapters.vdhand.hand_fk")),
        "hand_fk_end_site_scale": str(g("adapters.vdhand.hand_fk_end_site_scale")),
        "smooth": bool(g("policy.twist2.smooth")),
        "smooth_window_size": max(1, int(g("policy.twist2.smooth_window_size"))),
        "print_every": int(g("runtime.print_every")),
        "toggle_ramp_seconds": max(0.0, float(g("control.toggle_ramp_seconds"))),
        "exit_ramp_seconds": max(0.0, float(g("control.exit_ramp_seconds"))),
        "start_ramp_seconds": max(0.0, float(g("control.start_ramp_seconds"))),
        "evdev_device": str(g("control.evdev_device")),
        "evdev_grab": bool(g("control.evdev_grab")),
        "vmc_ip": str(g("adapters.slimevr.vmc_ip")),
        "vmc_port": int(g("adapters.slimevr.vmc_port")),
        "vmc_timeout_s": max(0.01, float(g("adapters.slimevr.vmc_timeout_s"))),
        "vmc_rot_mode": str(g("adapters.slimevr.vmc_rot_mode")),
        "vmc_use_fk": bool(g("adapters.slimevr.vmc_use_fk")),
        "vmc_use_viewer_fk": bool(g("adapters.slimevr.vmc_use_viewer_fk")),
        "vmc_fk_skeleton": str(g("adapters.slimevr.vmc_fk_skeleton")),
        "vmc_bvh_path": _resolve_repo_path(repo_root, str(g("adapters.slimevr.vmc_bvh_path"))),
        "vmc_bvh_scale": float(g("adapters.slimevr.vmc_bvh_scale")),
        "manus_address": str(g("adapters.manus.address")),
        "manus_left_sn": str(g("adapters.manus.left_sn")),
        "manus_right_sn": str(g("adapters.manus.right_sn")),
        "manus_auto_assign": bool(g("adapters.manus.auto_assign")),
        "manus_recv_timeout_ms": max(1, int(g("adapters.manus.recv_timeout_ms"))),
        "manus_flip_x": bool(g("adapters.manus.flip_x")),
        "max_steps": max(1, int(g("runtime.max_steps"))),
    }


def load_twist2_runtime_config(*, passthrough: list[str], repo_root: Path) -> Twist2RuntimeConfig:
    yaml_path, yaml_cfg = _load_yaml(passthrough, repo_root, "deploy_real/config/teleop.yaml")
    _validate_nested_schema(yaml_cfg, yaml_path)
    _assert_required_paths(yaml_cfg, _COMMON_REQUIRED_PATHS)
    twist2_policy = _get_nested(yaml_cfg, "policy.twist2")
    if not isinstance(twist2_policy, Mapping):
        raise ValueError("Config section 'policy.twist2' must be a mapping/dict")
    return Twist2RuntimeConfig(**_build_common_values(yaml_cfg, yaml_path, repo_root=repo_root))


def load_sonic_runtime_config(*, passthrough: list[str], repo_root: Path) -> SonicRuntimeConfig:
    yaml_path, yaml_cfg = _load_yaml(passthrough, repo_root, "deploy_real/config/teleop.yaml")
    _validate_nested_schema(yaml_cfg, yaml_path)
    _assert_required_paths(yaml_cfg, _COMMON_REQUIRED_PATHS + _SONIC_REQUIRED_PATHS)
    sonic_policy = _get_nested(yaml_cfg, "policy.sonic")
    if not isinstance(sonic_policy, Mapping):
        raise ValueError("Config section 'policy.sonic' must be a mapping/dict")
    common = _build_common_values(yaml_cfg, yaml_path, repo_root=repo_root)
    g = lambda p: _get_nested(yaml_cfg, p)
    return SonicRuntimeConfig(
        **common,
        zmq_bind_host=str(g("policy.sonic.zmq_bind_host")),
        zmq_pose_port=int(g("policy.sonic.zmq_pose_port")),
        zmq_pose_topic=str(g("policy.sonic.zmq_pose_topic")),
        zmq_num_frames_to_send=max(1, int(g("policy.sonic.zmq_num_frames_to_send"))),
        zmq_frame_index_step=int(g("policy.sonic.zmq_frame_index_step")),
        zmq_use_mailbox=bool(g("policy.sonic.zmq_use_mailbox")),
        zmq_catch_up=bool(g("policy.sonic.zmq_catch_up")),
    )

__all__ = [
    "SonicRuntimeConfig",
    "Twist2RuntimeConfig",
    "load_sonic_runtime_config",
    "load_twist2_runtime_config",
]
