#!/usr/bin/env python3
"""
CLI script to visualize predicted 35D body actions.

Usage:
    python visualize_body_actions.py \
        --actions predicted_actions.npy \
        --policy assets/ckpts/twist2_1017_20k.onnx \
        --xml assets/g1/g1_sim2sim_29dof.xml \
        --output visualization.mp4
"""

import argparse
import numpy as np
from pathlib import Path

from visualizers import HumanoidVisualizer, get_default_paths


def main():
    defaults = get_default_paths()
    
    parser = argparse.ArgumentParser(
        description='Visualize predicted 35D body actions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python visualize_body_actions.py \\
        --actions predicted_body.npy \\
        --output body_viz.mp4

    # Custom paths
    python visualize_body_actions.py \\
        --actions predicted_body.npy \\
        --policy /path/to/policy.onnx \\
        --xml /path/to/model.xml \\
        --output body_viz.mp4
        """
    )
    
    parser.add_argument('--actions', required=True,
                        help='Path to .npy file with (T, 35) actions')
    parser.add_argument('--policy', default=defaults["body_policy"],
                        help='Path to ONNX low-level RL policy')
    parser.add_argument('--xml', default=defaults["body_xml"],
                        help='MuJoCo XML model path')
    parser.add_argument('--output', default=None,
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS (default: 30)')
    parser.add_argument('--device', default='cpu',
                        help='Device for policy inference: cpu or cuda')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Simulation steps to stabilize at start pose (default: 100)')
    parser.add_argument('--width', type=int, default=640,
                        help='Rendering width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Rendering height (default: 480)')
    
    args = parser.parse_args()

    # Load actions
    print(f"Loading actions: {args.actions}")
    actions = np.load(args.actions)
    print(f"  Shape: {actions.shape}")

    # Validate
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[1] != 35:
        raise ValueError(f"Expected (T, 35) actions, got {actions.shape}")

    # Create visualizer
    viz = HumanoidVisualizer(
        xml_path=args.xml,
        policy_path=args.policy,
        device=args.device,
        width=args.width,
        height=args.height
    )

    # Visualize
    frames = viz.visualize(
        actions,
        output_video=args.output,
        fps=args.fps,
        warmup_steps=args.warmup_steps
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
