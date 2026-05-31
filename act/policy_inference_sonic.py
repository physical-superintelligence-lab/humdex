#!/usr/bin/env python3
"""
Token-level (sonic channel) online inference.

Loads an ACT checkpoint trained with `--channel sonic` (state_body 29 + action_token 64
+ hand), and drives the G1 through the GR00T C++ WBC controller:

  - body : 64-D encoder token  -> ZMQ Protocol-v4 `token_state` on tcp://*:5556 topic "pose"
  - hand : wuji qpos target    -> redis (action_wuji_qpos_target_*), consumed by server_wuji_hand
  - obs  : body state (29) from the controller's g1_debug stream (:5557, body_q_measured)
           + hand state (20/40) from redis + head camera

No temporal aggregation (tokens cannot be averaged). No safety ramp / safe-idle: a latent
token has no joint-space safe pose, so k/p only gate send/stop. VALIDATE IN SIM FIRST.

Prerequisite: the sonic controller (deploy.sh sim/real) must be running — it consumes the
token on 5556 and publishes body_q_measured on 5557.

Usage:
  python policy_inference_sonic.py --ckpt_dir DIR --hand_side both \
    --redis_ip <ip> --vision_ip <g1_ip> --state_zmq_ip <controller_ip>
"""

import os
import sys
import time
import argparse
from typing import Any, Dict, Optional

import numpy as np

# Match policy_inference.py path setup so `from policy import ACTPolicy` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.redis_util import RedisIO
from scripts.real_robot_util import VisionReader, KeyboardToggle
from scripts.policy_util import ACTPolicyWrapper

from policy_inference import Config  # reuse the inference config dataclass


# ---------------------------------------------------------------------------
# FSQ quantization
# ---------------------------------------------------------------------------
FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 0.0625


def fsq_quantize(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, FSQ_MIN, FSQ_MAX)
    return np.clip(np.round(clipped / FSQ_STEP) * FSQ_STEP, FSQ_MIN, FSQ_MAX)


# ---------------------------------------------------------------------------
# Body state reader: controller g1_debug stream (ZMQ SUB, msgpack, port 5557).
# Mirrors deploy_real/server_data_record.py::SonicStateZmqSource.
# ---------------------------------------------------------------------------
class SonicStateZmqSource:
    def __init__(self, ip: str, port: int = 5557, topic: str = "g1_debug"):
        import zmq  # type: ignore
        import msgpack  # type: ignore

        self._zmq = zmq
        self._msgpack = msgpack
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, str(topic))
        self._sock.connect(f"tcp://{ip}:{int(port)}")
        self._topic_bytes = str(topic).encode("utf-8")
        self._latest: Optional[Dict[str, Any]] = None

    def get_latest(self) -> Optional[Dict[str, Any]]:
        zmq = self._zmq
        while True:
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                break
            payload = raw[len(self._topic_bytes):] if raw.startswith(self._topic_bytes) else raw
            try:
                self._latest = self._msgpack.unpackb(payload, raw=False)
            except Exception:
                pass
        return self._latest

    def close(self):
        try:
            self._sock.close(0)
        except Exception:
            pass
        try:
            self._ctx.term()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Token publisher: ZMQ PUB, Protocol-v4 token_state -> C++ WBC (port 5556, "pose").
# ---------------------------------------------------------------------------
def _resolve_gear_sonic():
    """Import pack_pose_message / build_command_message from gear_sonic.

    GR00T-WholeBodyControl is expected next to the humdex repo (`../`) per the README;
    we also try inside the repo. Add it to sys.path so no PYTHONPATH is needed.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # humdex/
    candidates = [
        os.path.join(os.path.dirname(repo_root), "GR00T-WholeBodyControl"),  # ../GR00T-WholeBodyControl
        os.path.join(repo_root, "GR00T-WholeBodyControl"),                   # humdex/GR00T-WholeBodyControl
    ]
    for c in candidates:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
    try:
        m = __import__(
            "gear_sonic.utils.teleop.zmq.zmq_planner_sender",
            fromlist=["pack_pose_message", "build_command_message"],
        )
        return m.pack_pose_message, m.build_command_message
    except Exception as e:
        raise ImportError(
            "Could not import gear_sonic. Looked in: " + ", ".join(candidates)
            + ". Clone GR00T-WholeBodyControl next to humdex (../) per the README."
        ) from e


class TokenPublisher:
    def __init__(self, host: str = "*", port: int = 5556, topic: str = "pose"):
        import zmq  # type: ignore

        self._pack, self._cmd = _resolve_gear_sonic()
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{int(port)}")
        self._topic = topic
        time.sleep(0.3)  # let the controller's SUB connect before the first send

    def arm(self):
        self._sock.send(self._cmd(start=True, stop=False, planner=True))

    def stop(self):
        self._sock.send(self._cmd(start=False, stop=True, planner=False))

    def send_token(self, token_64: np.ndarray):
        payload = {"token_state": np.asarray(token_64, dtype=np.float32).reshape(1, 64)}
        self._sock.send(self._pack(payload, topic=self._topic, version=4))

    def close(self):
        try:
            self._sock.close(0)
        except Exception:
            pass
        try:
            self._ctx.term()
        except Exception:
            pass


# ---------------------------------------------------------------------------
def _read_hand_state(redis_io: RedisIO, config: Config) -> Optional[np.ndarray]:
    """Read wuji hand state (20 single / 40 both) from redis, body+hand training order."""
    def _get(key):
        raw = redis_io.client.get(key)
        v = redis_io._safe_json_load(raw)
        return None if v is None else np.asarray(v, dtype=np.float32).reshape(-1)

    if config.hand_side == "both":
        left = _get(config.key_state_hand_left)
        right = _get(config.key_state_hand_right)
        if left is None or right is None:
            return None
        return np.concatenate([left, right])
    side_key = config.key_state_hand_left if config.hand_side == "left" else config.key_state_hand_right
    return _get(side_key)


def eval_online_sonic(
    ckpt_dir: str,
    redis_ip: str = "localhost",
    redis_port: int = 6379,
    robot_key: str = "unitree_g1_with_hands",
    hand_side: str = "left",
    chunk_size: int = 50,
    frequency: float = 30.0,
    vision_ip: str = "192.168.123.164",
    vision_port: int = 5555,
    state_zmq_ip: str = "127.0.0.1",
    state_zmq_port: int = 5557,
    state_zmq_topic: str = "g1_debug",
    token_zmq_host: str = "*",
    token_zmq_port: int = 5556,
    max_timesteps: int = 1_000_000,
):
    config = Config(
        redis_ip=redis_ip, redis_port=redis_port, robot_key=robot_key,
        frequency=frequency, chunk_size=chunk_size, temporal_agg=False,
        hand_side=hand_side,
    )

    policy = ACTPolicyWrapper(ckpt_dir, config)
    redis_io = RedisIO(config)
    vision = VisionReader(server_ip=vision_ip, port=vision_port)
    state_src = SonicStateZmqSource(state_zmq_ip, state_zmq_port, state_zmq_topic)
    token_pub = TokenPublisher(token_zmq_host, token_zmq_port)
    kb = KeyboardToggle(enabled=True, toggle_send_key="k", hold_position_key="p")
    kb.start()

    use_both = (hand_side == "both")
    dt = 1.0 / float(frequency)
    armed = False

    print("\nSonic token inference running (k = send/stop, p = hold). Ctrl+C to quit.\n")
    try:
        for _t in range(int(max_timesteps)):
            t0 = time.time()
            live_image = vision.get_image()
            send_enabled, hold_enabled = kb.get()

            if send_enabled and (not hold_enabled):
                if not armed:
                    token_pub.arm()
                    armed = True

                st = state_src.get_latest()
                body_q = st.get("body_q_measured") if isinstance(st, dict) else None
                hand_state = _read_hand_state(redis_io, config)
                if body_q is None or hand_state is None:
                    time.sleep(dt)
                    continue

                qpos = np.concatenate([
                    np.asarray(body_q, dtype=np.float32).reshape(-1),  # 29
                    hand_state,                                        # 20 / 40
                ])
                action = policy(qpos, live_image)        # (64 + hand,)
                token = fsq_quantize(action[:64])
                hand = action[64:]

                token_pub.send_token(token)
                if use_both:
                    redis_io.publish_wuji_qpos_target(hand[:20], hand_side="left")
                    redis_io.publish_wuji_qpos_target(hand[20:], hand_side="right")
                else:
                    redis_io.publish_wuji_qpos_target(hand, hand_side=hand_side)
            else:
                # hold / default: stop streaming tokens (C++ WBC holds its last frame);
                # reset the chunk counter so we re-query fresh on re-arm.
                try:
                    policy.reset()
                except Exception:
                    pass

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            if armed:
                token_pub.stop()
        except Exception:
            pass
        try:
            kb.stop()
        except Exception:
            pass
        token_pub.close()
        state_src.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Token-level (sonic) online inference")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--redis_ip", default="localhost")
    p.add_argument("--redis_port", type=int, default=6379)
    p.add_argument("--robot_key", default="unitree_g1_with_hands")
    p.add_argument("--hand_side", default="left", choices=["left", "right", "both"])
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--frequency", type=float, default=30.0)
    p.add_argument("--vision_ip", default="192.168.123.164")
    p.add_argument("--vision_port", type=int, default=5555)
    p.add_argument("--state_zmq_ip", default="127.0.0.1")
    p.add_argument("--state_zmq_port", type=int, default=5557)
    p.add_argument("--state_zmq_topic", default="g1_debug")
    p.add_argument("--token_zmq_host", default="*")
    p.add_argument("--token_zmq_port", type=int, default=5556)
    eval_online_sonic(**vars(p.parse_args()))
