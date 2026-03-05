from __future__ import annotations

from typing import Dict, List

from deploy_real.pipelines.utils.config import load_sonic_runtime_config
from deploy_real.pipelines.utils.stages import (
    sonic_cleanup_components,
    sonic_stage_build_hand_tracking,
    sonic_stage_init_components,
    sonic_stage_publish_body_zmq,
    sonic_stage_publish_hand_redis,
    sonic_stage_retarget_body,
)
from deploy_real.pipelines.pipeline import ComponentSelection, OrchestratedPipeline


PIPELINE_STAGES = (
    "load_config",
    "init_components",
    "read_body_frame",
    "read_hand_frame",
    "apply_keyboard_state",
    "retarget_body",
    "build_hand_tracking",
    "publish_zmq_pose",
    "publish_hand_redis",
    "step_rate_limiter",
)


def _resolve_body_adapter(body_source: str) -> str:
    mapping = {
        "vdmocap": "deploy_real.adapters.body.vdmocap_adapter",
        "slimevr": "deploy_real.adapters.body.slimevr_adapter",
    }
    if body_source not in mapping:
        raise ValueError(f"Unsupported body_source for sonic pipeline: {body_source}")
    return mapping[body_source]


def _resolve_hand_adapter(hand_source: str) -> str:
    mapping = {
        "vdhand": "deploy_real.adapters.hand.vdhand_adapter",
        "manus": "deploy_real.adapters.hand.manus_adapter",
    }
    if hand_source not in mapping:
        raise ValueError(f"Unsupported hand_source for sonic pipeline: {hand_source}")
    return mapping[hand_source]


def build_pipeline(body_source: str, hand_source: str, passthrough: List[str] | None = None) -> OrchestratedPipeline:
    selection = ComponentSelection(
        body_adapter=_resolve_body_adapter(body_source),
        hand_adapter=_resolve_hand_adapter(hand_source),
        policy_core="sonic_policy_core",
        publisher="redis_publisher",
        zmq_publisher="zmq_pose_publisher",
    )
    return OrchestratedPipeline(
        policy="sonic",
        selection=selection,
        passthrough=passthrough or [],
        stages=PIPELINE_STAGES,
        load_config_fn=load_sonic_runtime_config,
        init_components_fn=sonic_stage_init_components,
        retarget_body_fn=sonic_stage_retarget_body,
        build_hand_tracking_fn=sonic_stage_build_hand_tracking,
        publish_fns=[sonic_stage_publish_body_zmq, sonic_stage_publish_hand_redis],
        cleanup_fn=sonic_cleanup_components,
        config_fields=[
            "dst_ip",
            "dst_port",
            "target_fps",
            "hands",
            "keyboard_backend",
            "zmq_bind_host",
            "zmq_pose_port",
            "zmq_pose_topic",
        ],
        component_fields=[
            "sdk_ready",
            "body_sdk_ready",
            "hand_sdk_ready",
            "gmr_ready",
            "zmq_ready",
        ],
        status_fields=[
            "body_frame_status",
            "hand_frame_status",
            "retarget_status",
            "zmq_publish_status",
            "redis_hand_status",
        ],
    )


def describe_pipeline(body_source: str, hand_source: str, passthrough: List[str] | None = None) -> Dict[str, object]:
    return build_pipeline(body_source=body_source, hand_source=hand_source, passthrough=passthrough).describe()
