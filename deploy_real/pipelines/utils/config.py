from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


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
    measure_fps: int
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
    publish_bvh_hand: bool
    hand_fk: bool
    hand_fk_end_site_scale: str
    control_neck: bool
    neck_retarget_scale: float
    control_gripper_hand_action: bool
    pinch_mode: bool
    hand_step: float
    smooth: bool
    smooth_window_size: int
    print_every: int
    dry_run: bool
    toggle_ramp_seconds: float
    exit_ramp_seconds: float
    start_ramp_seconds: float
    evdev_device: str
    evdev_grab: bool
    vmc_ip: str
    vmc_port: int
    vmc_timeout_s: float
    vmc_rot_mode: str
    vmc_invert_zw: bool
    vmc_use_fk: bool
    vmc_use_viewer_fk: bool
    vmc_fk_skeleton: str
    vmc_bvh_path: str
    vmc_bvh_scale: float
    vmc_viewer_bone_axis_override: str
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
    enable_zmq_pose: bool
    zmq_bind_host: str
    zmq_pose_port: int
    zmq_pose_topic: str
    zmq_num_frames_to_send: int
    zmq_frame_index_step: int
    zmq_use_mailbox: bool
    zmq_heading_increment: float
    zmq_catch_up: bool
    zmq_include_hand_joints: bool


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


def _build_common_values(yaml_cfg: Dict[str, Any], yaml_path: Path, *, defaults: Dict[str, Any]) -> Dict[str, Any]:
    g = lambda k: yaml_cfg.get(k, defaults[k])
    return {
        "config_yaml": str(yaml_path),
        "redis_ip": str(g("redis_ip")),
        "dst_ip": str(g("dst_ip")),
        "dst_port": int(g("dst_port")),
        "local_port": int(g("local_port")),
        "mocap_index": int(g("mocap_index")),
        "world_space": int(g("world_space")),
        "dst_ip_body": str(g("dst_ip_body")),
        "dst_port_body": int(g("dst_port_body")),
        "mocap_index_body": int(g("mocap_index_body")),
        "dst_ip_hand": str(g("dst_ip_hand")),
        "dst_port_hand": int(g("dst_port_hand")),
        "mocap_index_hand": int(g("mocap_index_hand")),
        "actual_human_height": float(g("actual_human_height")),
        "target_fps": float(g("target_fps")),
        "measure_fps": int(g("measure_fps")),
        "hands": str(g("hands")),
        "format": str(g("format")),
        "offset_to_ground": bool(g("offset_to_ground")),
        "toggle_send_key": str(g("toggle_send_key")),
        "hold_position_key": str(g("hold_position_key")),
        "safe_idle_pose_id": str(g("safe_idle_pose_id")),
        "ramp_ease": str(g("ramp_ease")),
        "keyboard_toggle_send": bool(g("keyboard_toggle_send")),
        "keyboard_backend": str(g("keyboard_backend")).strip().lower(),
        "hand_source_timeout_s": float(g("hand_source_timeout_s")),
        "publish_bvh_hand": bool(g("publish_bvh_hand")),
        "hand_fk": bool(g("hand_fk")),
        "hand_fk_end_site_scale": str(g("hand_fk_end_site_scale")),
        "control_neck": bool(g("control_neck")),
        "neck_retarget_scale": float(g("neck_retarget_scale")),
        "control_gripper_hand_action": bool(g("control_gripper_hand_action")),
        "pinch_mode": bool(g("pinch_mode")),
        "hand_step": float(g("hand_step")),
        "smooth": bool(g("smooth")),
        "smooth_window_size": max(1, int(g("smooth_window_size"))),
        "print_every": int(g("print_every")),
        "dry_run": bool(g("dry_run")),
        "toggle_ramp_seconds": max(0.0, float(g("toggle_ramp_seconds"))),
        "exit_ramp_seconds": max(0.0, float(g("exit_ramp_seconds"))),
        "start_ramp_seconds": max(0.0, float(g("start_ramp_seconds"))),
        "evdev_device": str(g("evdev_device")),
        "evdev_grab": bool(g("evdev_grab")),
        "vmc_ip": str(g("vmc_ip")),
        "vmc_port": int(g("vmc_port")),
        "vmc_timeout_s": max(0.01, float(g("vmc_timeout_s"))),
        "vmc_rot_mode": str(g("vmc_rot_mode")),
        "vmc_invert_zw": bool(g("vmc_invert_zw")),
        "vmc_use_fk": bool(g("vmc_use_fk")),
        "vmc_use_viewer_fk": bool(g("vmc_use_viewer_fk")),
        "vmc_fk_skeleton": str(g("vmc_fk_skeleton")),
        "vmc_bvh_path": str(g("vmc_bvh_path")),
        "vmc_bvh_scale": float(g("vmc_bvh_scale")),
        "vmc_viewer_bone_axis_override": str(g("vmc_viewer_bone_axis_override")),
        "manus_address": str(g("manus_address")),
        "manus_left_sn": str(g("manus_left_sn")),
        "manus_right_sn": str(g("manus_right_sn")),
        "manus_auto_assign": bool(g("manus_auto_assign")),
        "manus_recv_timeout_ms": max(1, int(g("manus_recv_timeout_ms"))),
        "manus_flip_x": bool(g("manus_flip_x")),
        "max_steps": max(1, int(g("max_steps"))),
    }


def load_twist2_runtime_config(*, passthrough: list[str], repo_root: Path) -> Twist2RuntimeConfig:
    yaml_path, yaml_cfg = _load_yaml(passthrough, repo_root, "deploy_real/config/teleop_twist2_vdmocap_vdhand.yaml")
    if len(yaml_cfg) == 0:
        raise ValueError(f"Config yaml is empty or invalid: {yaml_path}")
    defaults = {
        "redis_ip": "localhost",
        "dst_ip": "192.168.1.112",
        "dst_port": 7000,
        "local_port": 0,
        "mocap_index": 0,
        "world_space": 0,
        "dst_ip_body": "",
        "dst_port_body": 0,
        "mocap_index_body": -1,
        "dst_ip_hand": "",
        "dst_port_hand": 0,
        "mocap_index_hand": -1,
        "actual_human_height": 1.45,
        "target_fps": 0.0,
        "measure_fps": 0,
        "hands": "both",
        "format": "nokov",
        "offset_to_ground": False,
        "toggle_send_key": "k",
        "hold_position_key": "p",
        "safe_idle_pose_id": "0",
        "ramp_ease": "cosine",
        "keyboard_toggle_send": False,
        "keyboard_backend": "stdin",
        "hand_source_timeout_s": 0.5,
        "publish_bvh_hand": False,
        "hand_fk": False,
        "hand_fk_end_site_scale": "0.8",
        "control_neck": False,
        "neck_retarget_scale": 1.5,
        "control_gripper_hand_action": False,
        "pinch_mode": False,
        "hand_step": 0.05,
        "smooth": False,
        "smooth_window_size": 5,
        "print_every": 120,
        "dry_run": False,
        "toggle_ramp_seconds": 0.0,
        "exit_ramp_seconds": 0.0,
        "start_ramp_seconds": 0.0,
        "evdev_device": "auto",
        "evdev_grab": False,
        "vmc_ip": "0.0.0.0",
        "vmc_port": 39539,
        "vmc_timeout_s": 0.5,
        "vmc_rot_mode": "local",
        "vmc_invert_zw": False,
        "vmc_use_fk": True,
        "vmc_use_viewer_fk": True,
        "vmc_fk_skeleton": "bvh",
        "vmc_bvh_path": "bvh-recording.bvh",
        "vmc_bvh_scale": 0.01,
        "vmc_viewer_bone_axis_override": "Hips:flip=yz;Spine:flip=yz;Spine1:flip=yz;Neck:flip=yz;Head:flip=yz;LeftUpperArm:flip=x;RightUpperArm:flip=x;LeftLowerArm:flip=x;RightLowerArm:flip=x;LeftHand:flip=x;RightHand:flip=x;LeftUpperLeg:flip=x;RightUpperLeg:flip=x;LeftLowerLeg:flip=x;RightLowerLeg:flip=x;LeftFoot:flip=x;RightFoot:flip=x",
        "manus_address": "tcp://127.0.0.1:7668",
        "manus_left_sn": "",
        "manus_right_sn": "",
        "manus_auto_assign": True,
        "manus_recv_timeout_ms": 20,
        "manus_flip_x": False,
        "max_steps": 1,
    }
    return Twist2RuntimeConfig(**_build_common_values(yaml_cfg, yaml_path, defaults=defaults))


def load_sonic_runtime_config(*, passthrough: list[str], repo_root: Path) -> SonicRuntimeConfig:
    yaml_path, yaml_cfg = _load_yaml(passthrough, repo_root, "deploy_real/config/teleop_sonic_vdmocap_vdhand.yaml")
    defaults = {
        "redis_ip": "localhost",
        "dst_ip": "192.168.1.111",
        "dst_port": 7000,
        "local_port": 0,
        "mocap_index": 0,
        "world_space": 0,
        "dst_ip_body": "",
        "dst_port_body": 0,
        "mocap_index_body": -1,
        "dst_ip_hand": "",
        "dst_port_hand": 0,
        "mocap_index_hand": -1,
        "actual_human_height": 1.45,
        "target_fps": 50.0,
        "measure_fps": 1,
        "hands": "both",
        "format": "nokov",
        "offset_to_ground": True,
        "toggle_send_key": "k",
        "hold_position_key": "p",
        "safe_idle_pose_id": "2",
        "ramp_ease": "cosine",
        "keyboard_toggle_send": True,
        "keyboard_backend": "stdin",
        "hand_source_timeout_s": 0.5,
        "publish_bvh_hand": True,
        "hand_fk": True,
        "hand_fk_end_site_scale": "0.858,0.882,0.882,0.882,0.882",
        "control_neck": False,
        "neck_retarget_scale": 1.5,
        "control_gripper_hand_action": False,
        "pinch_mode": False,
        "hand_step": 0.05,
        "smooth": False,
        "smooth_window_size": 5,
        "print_every": 120,
        "dry_run": False,
        "toggle_ramp_seconds": 3.0,
        "exit_ramp_seconds": 3.0,
        "start_ramp_seconds": 3.0,
        "evdev_device": "auto",
        "evdev_grab": False,
        "vmc_ip": "0.0.0.0",
        "vmc_port": 39539,
        "vmc_timeout_s": 0.5,
        "vmc_rot_mode": "local",
        "vmc_invert_zw": False,
        "vmc_use_fk": True,
        "vmc_use_viewer_fk": True,
        "vmc_fk_skeleton": "bvh",
        "vmc_bvh_path": "bvh-recording.bvh",
        "vmc_bvh_scale": 0.01,
        "vmc_viewer_bone_axis_override": "Hips:flip=yz;Spine:flip=yz;Spine1:flip=yz;Neck:flip=yz;Head:flip=yz;LeftUpperArm:flip=x;RightUpperArm:flip=x;LeftLowerArm:flip=x;RightLowerArm:flip=x;LeftHand:flip=x;RightHand:flip=x;LeftUpperLeg:flip=x;RightUpperLeg:flip=x;LeftLowerLeg:flip=x;RightLowerLeg:flip=x;LeftFoot:flip=x;RightFoot:flip=x",
        "manus_address": "tcp://127.0.0.1:7668",
        "manus_left_sn": "",
        "manus_right_sn": "",
        "manus_auto_assign": True,
        "manus_recv_timeout_ms": 20,
        "manus_flip_x": False,
        "max_steps": 1,
    }
    common = _build_common_values(yaml_cfg, yaml_path, defaults=defaults)
    g = lambda k, d: yaml_cfg.get(k, d)
    return SonicRuntimeConfig(
        **common,
        enable_zmq_pose=bool(g("enable_zmq_pose", True)),
        zmq_bind_host=str(g("zmq_bind_host", "0.0.0.0")),
        zmq_pose_port=int(g("zmq_pose_port", 5556)),
        zmq_pose_topic=str(g("zmq_pose_topic", "pose")),
        zmq_num_frames_to_send=max(1, int(g("zmq_num_frames_to_send", 1))),
        zmq_frame_index_step=int(g("zmq_frame_index_step", 0)),
        zmq_use_mailbox=bool(g("zmq_use_mailbox", True)),
        zmq_heading_increment=float(g("zmq_heading_increment", 0.0)),
        zmq_catch_up=bool(g("zmq_catch_up", True)),
        zmq_include_hand_joints=bool(g("zmq_include_hand_joints", True)),
    )

__all__ = [
    "SonicRuntimeConfig",
    "Twist2RuntimeConfig",
    "load_sonic_runtime_config",
    "load_twist2_runtime_config",
]
