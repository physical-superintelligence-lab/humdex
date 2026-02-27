from __future__ import annotations

from typing import Dict, List

from deploy_real.pipelines.utils.config import load_twist2_runtime_config
from deploy_real.pipelines.utils.stages import (
    twist2_cleanup_components,
    twist2_stage_build_hand_tracking,
    twist2_stage_init_components,
    twist2_stage_publish_redis,
    twist2_stage_retarget_body,
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
    "publish_redis",
    "step_rate_limiter",
)


def _resolve_body_adapter(body_source: str) -> str:
    mapping = {
        "vdmocap": "deploy_real.adapters.body.vdmocap_adapter",
        "slimevr": "deploy_real.adapters.body.slimevr_adapter",
    }
    if body_source not in mapping:
        raise ValueError(f"Unsupported body_source for twist2 pipeline: {body_source}")
    return mapping[body_source]


def _resolve_hand_adapter(hand_source: str) -> str:
    mapping = {
        "vdhand": "deploy_real.adapters.hand.vdhand_adapter",
        "manus": "deploy_real.adapters.hand.manus_adapter",
    }
    if hand_source not in mapping:
        raise ValueError(f"Unsupported hand_source for twist2 pipeline: {hand_source}")
    return mapping[hand_source]


def build_pipeline(body_source: str, hand_source: str, passthrough: List[str] | None = None) -> OrchestratedPipeline:
    selection = ComponentSelection(
        body_adapter=_resolve_body_adapter(body_source),
        hand_adapter=_resolve_hand_adapter(hand_source),
        policy_core="twist2_policy_core",
        publisher="redis_publisher",
    )
    return OrchestratedPipeline(
        policy="twist2",
        selection=selection,
        passthrough=passthrough or [],
        stages=PIPELINE_STAGES,
        load_config_fn=load_twist2_runtime_config,
        init_components_fn=twist2_stage_init_components,
        retarget_body_fn=twist2_stage_retarget_body,
        build_hand_tracking_fn=twist2_stage_build_hand_tracking,
        publish_fns=[twist2_stage_publish_redis],
        cleanup_fn=twist2_cleanup_components,
        config_fields=[
            "dst_ip",
            "dst_port",
            "redis_ip",
            "target_fps",
            "hands",
            "keyboard_backend",
            "safe_idle_pose_id",
        ],
        component_fields=[
            "redis_connected",
            "sdk_ready",
            "body_sdk_ready",
            "hand_sdk_ready",
            "gmr_ready",
        ],
        status_fields=[
            "body_frame_status",
            "hand_frame_status",
            "retarget_status",
            "publish_status",
        ],
    )


def describe_pipeline(body_source: str, hand_source: str, passthrough: List[str] | None = None) -> Dict[str, object]:
    return build_pipeline(body_source=body_source, hand_source=hand_source, passthrough=passthrough).describe()
