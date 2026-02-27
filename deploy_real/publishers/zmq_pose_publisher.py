from __future__ import annotations

from typing import Any, Dict

import zmq  # type: ignore


def create_zmq_pose_publisher(
    *,
    bind_host: str,
    port: int,
    topic: str,
    pack_pose_message: Any,
) -> Dict[str, Any]:
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    addr = f"tcp://{str(bind_host)}:{int(port)}"
    socket.bind(addr)
    return {
        "ctx": ctx,
        "socket": socket,
        "addr": addr,
        "topic": str(topic),
        "pack_pose_message": pack_pose_message,
    }


def publish_zmq_pose_step(*, publisher: Dict[str, Any], payload: Dict[str, Any]) -> None:
    pack_pose_message = publisher.get("pack_pose_message", None)
    socket = publisher.get("socket", None)
    topic = str(publisher.get("topic", "pose"))
    if pack_pose_message is None or socket is None:
        return
    packed = pack_pose_message(payload, topic=topic, version=1)  # type: ignore[misc]
    socket.send(packed)


def close_zmq_pose_publisher(publisher: Dict[str, Any] | None) -> None:
    if not isinstance(publisher, dict):
        return
    socket = publisher.get("socket", None)
    if socket is not None:
        try:
            socket.close(0)
        except Exception:
            pass
