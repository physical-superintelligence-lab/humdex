from __future__ import annotations

from typing import Any, Dict, Iterable


_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"
_ANSI_CYAN = "\033[36m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"


def _colorize_value(v: Any) -> str:
    s = str(v)
    low = s.lower()
    if ("error" in low) or ("fail" in low) or ("exception" in low):
        return f"{_ANSI_RED}{s}{_ANSI_RESET}"
    if ("ok" in low) or (low == "true"):
        return f"{_ANSI_GREEN}{s}{_ANSI_RESET}"
    if ("no_update" in low) or ("unavailable" in low) or ("disabled" in low):
        return f"{_ANSI_YELLOW}{s}{_ANSI_RESET}"
    return f"{_ANSI_DIM}{s}{_ANSI_RESET}"


def _prefix(policy: str) -> str:
    return f"{_ANSI_CYAN}[{policy}_pipeline]{_ANSI_RESET}"


def log_pipeline_start(
    *,
    policy: str,
    cfg: Any,
    state: Any,
    comps: Dict[str, Any],
    config_fields: Iterable[str],
    component_fields: Iterable[str],
) -> None:
    prefix = _prefix(policy)
    print(f"{prefix} stage progress: load_config -> init_components -> loop(staged)")
    cfg_parts = [f"{name}={getattr(cfg, name)}" for name in config_fields]
    print(f"{prefix} config " + " ".join(cfg_parts))
    src = state.context.get("resolved_sources", {})
    if isinstance(src, dict):
        if bool(src.get("split_sources", False)):
            print(
                f"{prefix} sources "
                f"body={src.get('dst_ip_body')}:{src.get('dst_port_body')} idx={src.get('mocap_index_body')} "
                f"hand={src.get('dst_ip_hand')}:{src.get('dst_port_hand')} idx={src.get('mocap_index_hand')} "
                "split_sources=True"
            )
        else:
            print(
                f"{prefix} sources "
                f"unified={src.get('dst_ip_body')}:{src.get('dst_port_body')} idx={src.get('mocap_index_body')} "
                "split_sources=False"
            )
    comp_parts = [f"{name}={bool(comps.get(name, False))}" for name in component_fields]
    print(f"{prefix} components " + " ".join(comp_parts))
    for key in ["redis_error", "sdk_error", "hand_sdk_error", "gmr_error", "zmq_error"]:
        if key in comps:
            print(f"{prefix} {key}={comps[key]}")


def log_step(*, policy: str, state: Any, status_fields: Iterable[str]) -> None:
    cfg = state.context.get("config", None)
    print_every = max(1, int(getattr(cfg, "print_every", 120))) if cfg is not None else 120
    signature = tuple(state.context.get(name) for name in status_fields)
    prev_signature = state.context.get("_last_log_signature", None)
    changed = signature != prev_signature
    should_print = changed or (state.loop_count <= 3) or (state.loop_count % print_every == 0)
    if not should_print:
        return
    state.context["_last_log_signature"] = signature
    prefix = _prefix(policy)
    parts = [f"{name}={_colorize_value(state.context.get(name))}" for name in status_fields]
    print(
        f"{prefix} step {state.loop_count} frame_index={state.frame_index} "
        f"hand_frame_index={getattr(state, 'hand_frame_index', -1)} "
        + " ".join(parts)
    )


def log_periodic(*, frame_index: int, step: int, now_ms: int) -> None:
    print("[teleop] " f"frameIndex={frame_index} step={step} t_action_ms={now_ms}")


def log_exit_requested(*, policy: str) -> None:
    print(f"[{policy}_pipeline] exit requested by keyboard.")


def log_exit_ramp_finished(*, policy: str) -> None:
    print(f"[{policy}_pipeline] exit ramp finished.")
