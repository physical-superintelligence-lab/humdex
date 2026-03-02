#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from deploy_real.pipelines.pipeline import build_pipeline_instance, resolve_pipeline_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified teleop entry (router-only v1).")
    parser.add_argument("--policy", choices=["twist2", "sonic"], default="twist2")
    parser.add_argument("--body_source", "--body", dest="body_source", choices=["vdmocap", "slimevr"], default="vdmocap")
    parser.add_argument("--hand_source", "--hand", dest="hand_source", choices=["vdhand", "manus"], default="vdhand")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Arguments after '--' will be forwarded.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    passthrough = list(args.passthrough)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    spec = resolve_pipeline_spec(
        policy=args.policy,
        body_source=args.body_source,
        hand_source=args.hand_source,
        passthrough=passthrough,
    )

    print(f"[teleop] policy={args.policy} body_source={args.body_source} hand_source={args.hand_source}")
    print(f"[teleop] pipeline={spec['policy']}")
    print(f"[teleop] stages={','.join(spec['stages'])}")
    components = spec["components"]
    print("[teleop] components:")
    print(f"  - body_adapter: {components['body_adapter']}")
    print(f"  - hand_adapter: {components['hand_adapter']}")
    print(f"  - policy_core: {components['policy_core']}")
    print(f"  - publisher: {components['publisher']}")
    if "zmq_publisher" in components:
        print(f"  - zmq_publisher: {components['zmq_publisher']}")
    print(f"  - keyboard: {components['keyboard']}")
    print(f"  - rate_limiter: {components['rate_limiter']}")
    if passthrough:
        print(f"[teleop] passthrough={passthrough}")

    pipeline = build_pipeline_instance(
        policy=args.policy,
        body_source=args.body_source,
        hand_source=args.hand_source,
        passthrough=passthrough,
    )
    return int(pipeline.run())


if __name__ == "__main__":
    sys.exit(main())

