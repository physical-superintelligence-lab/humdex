#!/usr/bin/env python3
"""
Analyze the action-to-state tracking delay in robot episodes.

For each action at timestep t, we compute the distance to future states (t+1, t+2, ..., t+max_k)
and find the k that minimizes the distance. This helps understand how many timesteps it takes
for the robot to reach the commanded action.

Analyzed pairs:
1. body: state_body[3:34] vs action_body[3:5] + action_body[6:35] (31 dims)
2. hand_left: state_wuji_hand_left vs action_wuji_qpos_target_left (20 dims)
3. hand_right: state_wuji_hand_right vs action_wuji_qpos_target_right (20 dims)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


# Define the analysis targets
ANALYSIS_TARGETS = {
    'body': {
        'action_key': 'action_body',
        'state_key': 'state_body',
        'action_extractor': lambda x: x[3:5] + x[6:35],  # 31 dims
        'state_extractor': lambda x: x[3:34],  # 31 dims
        'description': 'Body (31 dims)'
    },
    'hand_left': {
        'action_key': 'action_wuji_qpos_target_left',
        'state_key': 'state_wuji_hand_left',
        'action_extractor': lambda x: x,  # all 20 dims
        'state_extractor': lambda x: x,   # all 20 dims
        'description': 'Left Hand (20 dims)'
    },
    'hand_right': {
        'action_key': 'action_wuji_qpos_target_right',
        'state_key': 'state_wuji_hand_right',
        'action_extractor': lambda x: x,  # all 20 dims
        'state_extractor': lambda x: x,   # all 20 dims
        'description': 'Right Hand (20 dims)'
    }
}


def extract_state_dims(state_body: list) -> np.ndarray:
    """Extract the 31 comparable dimensions from state_body."""
    return np.array(state_body[3:34])


def extract_action_dims(action_body: list) -> np.ndarray:
    """Extract the 31 comparable dimensions from action_body."""
    # action_body[3:5] (2 dims) + action_body[6:35] (29 dims) = 31 dims
    part1 = action_body[3:5]   # indices 3, 4
    part2 = action_body[6:35]  # indices 6 to 34
    return np.array(part1 + part2)


def compute_distance(a: np.ndarray, b: np.ndarray, metric: str = 'l2') -> float:
    """Compute distance between two vectors."""
    if metric == 'l2':
        return np.sqrt(np.sum((a - b) ** 2))
    elif metric == 'l1':
        return np.sum(np.abs(a - b))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def analyze_episode_for_target(frames: list, target_name: str, max_k: int = 30, metrics: list = ['l2', 'l1']) -> dict:
    """
    Analyze a single episode for a specific target (body, hand_left, hand_right).
    
    Args:
        frames: List of frame data
        target_name: Name of the target ('body', 'hand_left', 'hand_right')
        max_k: Maximum lookahead steps to consider
        metrics: List of distance metrics to use ('l1', 'l2')
    
    Returns:
        Dictionary with results for each metric
    """
    target_config = ANALYSIS_TARGETS[target_name]
    action_key = target_config['action_key']
    state_key = target_config['state_key']
    action_extractor = target_config['action_extractor']
    state_extractor = target_config['state_extractor']
    
    num_frames = len(frames)
    results = {metric: {
        'best_k': [], 
        'best_distance': [], 
        'all_distances': [],
        # New: store distances per k for global analysis
        'distances_per_k': {k: [] for k in range(1, max_k + 1)}
    } for metric in metrics}
    
    for t in range(num_frames):
        frame = frames[t]
        
        # Skip if action is missing
        action_data = frame.get(action_key)
        if action_data is None:
            continue
            
        action = np.array(action_extractor(action_data))
        
        for metric in metrics:
            distances_for_k = []
            
            # Compare action at t to states at t+1, t+2, ..., t+max_k
            for k in range(1, min(max_k + 1, num_frames - t)):
                future_frame = frames[t + k]
                
                # Skip if state is missing
                state_data = future_frame.get(state_key)
                if state_data is None:
                    distances_for_k.append(float('inf'))
                    continue
                
                state = np.array(state_extractor(state_data))
                dist = compute_distance(action, state, metric)
                distances_for_k.append(dist)
                
                # Store distance for this k (for global analysis)
                results[metric]['distances_per_k'][k].append(dist)
            
            if len(distances_for_k) > 0 and min(distances_for_k) < float('inf'):
                best_k = np.argmin(distances_for_k) + 1  # +1 because k starts from 1
                best_dist = min(distances_for_k)
                
                results[metric]['best_k'].append(best_k)
                results[metric]['best_distance'].append(best_dist)
                results[metric]['all_distances'].append(distances_for_k)
    
    return results


def analyze_episode(data_path: Path, max_k: int = 30, metrics: list = ['l2', 'l1'], targets: list = None) -> dict:
    """
    Analyze a single episode to find the best k for each action frame.
    
    Args:
        data_path: Path to data.json
        max_k: Maximum lookahead steps to consider
        metrics: List of distance metrics to use ('l1', 'l2')
        targets: List of targets to analyze ('body', 'hand_left', 'hand_right')
    
    Returns:
        Dictionary with results for each target and metric
    """
    if targets is None:
        targets = list(ANALYSIS_TARGETS.keys())
    
    with open(data_path, 'r') as f:
        episode_data = json.load(f)
    
    frames = episode_data['data']
    
    results = {}
    for target_name in targets:
        results[target_name] = analyze_episode_for_target(frames, target_name, max_k, metrics)
    
    return results


def analyze_all_episodes(base_dir: Path, max_k: int = 30, metrics: list = ['l2', 'l1'], targets: list = None) -> dict:
    """Analyze all episodes in the directory."""
    
    if targets is None:
        targets = list(ANALYSIS_TARGETS.keys())
    
    # Find all episode directories
    episode_dirs = sorted([d for d in base_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('episode_')])
    
    print(f"Found {len(episode_dirs)} episodes to analyze")
    print(f"Targets: {targets}")
    
    # Initialize results structure
    all_results = {}
    for target_name in targets:
        all_results[target_name] = {
            metric: {
                'best_k': [], 
                'best_distance': [], 
                'per_episode': {},
                'distances_per_k': {k: [] for k in range(1, max_k + 1)}
            } 
            for metric in metrics
        }
    
    for ep_dir in episode_dirs:
        data_path = ep_dir / 'data.json'
        if not data_path.exists():
            print(f"  Skipping {ep_dir.name}: no data.json found")
            continue
        
        print(f"  Analyzing {ep_dir.name}...")
        ep_results = analyze_episode(data_path, max_k, metrics, targets)
        
        for target_name in targets:
            for metric in metrics:
                target_ep_results = ep_results[target_name][metric]
                if len(target_ep_results['best_k']) > 0:
                    all_results[target_name][metric]['best_k'].extend(target_ep_results['best_k'])
                    all_results[target_name][metric]['best_distance'].extend(target_ep_results['best_distance'])
                    all_results[target_name][metric]['per_episode'][ep_dir.name] = {
                        'mean_k': np.mean(target_ep_results['best_k']),
                        'median_k': np.median(target_ep_results['best_k']),
                        'std_k': np.std(target_ep_results['best_k']),
                        'mean_dist': np.mean(target_ep_results['best_distance']),
                        'num_frames': len(target_ep_results['best_k'])
                    }
                    
                    # Aggregate distances per k
                    for k in range(1, max_k + 1):
                        all_results[target_name][metric]['distances_per_k'][k].extend(
                            target_ep_results['distances_per_k'][k]
                        )
    
    return all_results


def get_mode(arr):
    """Get the mode (most frequent value) of an array."""
    values, counts = np.unique(arr, return_counts=True)
    mode_idx = np.argmax(counts)
    return values[mode_idx], counts[mode_idx]


def print_summary(results: dict, metrics: list, targets: list):
    """Print a simplified comparison table."""
    
    print("\n" + "=" * 90)
    print("ACTION DELAY ANALYSIS - COMPARISON TABLE")
    print("=" * 90)
    
    # Collect data for all targets and metrics
    table_data = []
    
    for target_name in targets:
        target_results = results[target_name]
        target_desc = ANALYSIS_TARGETS[target_name]['description']
        
        for metric in metrics:
            metric_results = target_results[metric]
            best_k = np.array(metric_results['best_k'])
            
            if len(best_k) == 0:
                table_data.append({
                    'target': target_desc,
                    'metric': metric.upper(),
                    'frames': 0,
                    'mode_1st': '-', 'mode_1st_pct': '-',
                    'mode_2nd': '-', 'mode_2nd_pct': '-',
                    'global_1st': '-', 'global_1st_err': '-',
                    'global_2nd': '-', 'global_2nd_err': '-',
                })
                continue
            
            # Method 1: Per-frame mode (top 2)
            values, counts = np.unique(best_k, return_counts=True)
            sorted_indices = np.argsort(-counts)  # descending
            mode_1st_k = values[sorted_indices[0]]
            mode_1st_pct = 100 * counts[sorted_indices[0]] / len(best_k)
            if len(sorted_indices) > 1:
                mode_2nd_k = values[sorted_indices[1]]
                mode_2nd_pct = 100 * counts[sorted_indices[1]] / len(best_k)
            else:
                mode_2nd_k, mode_2nd_pct = '-', '-'
            
            # Method 2: Global mean error (top 2)
            distances_per_k = metric_results.get('distances_per_k', {})
            k_errors = []
            for k, dists in distances_per_k.items():
                if len(dists) > 0:
                    k_errors.append((k, np.mean(dists)))
            k_errors_sorted = sorted(k_errors, key=lambda x: x[1])
            
            if len(k_errors_sorted) >= 1:
                global_1st_k, global_1st_err = k_errors_sorted[0]
            else:
                global_1st_k, global_1st_err = '-', '-'
            
            if len(k_errors_sorted) >= 2:
                global_2nd_k, global_2nd_err = k_errors_sorted[1]
            else:
                global_2nd_k, global_2nd_err = '-', '-'
            
            table_data.append({
                'target': target_desc,
                'metric': metric.upper(),
                'frames': len(best_k),
                'mode_1st': mode_1st_k, 'mode_1st_pct': mode_1st_pct,
                'mode_2nd': mode_2nd_k, 'mode_2nd_pct': mode_2nd_pct,
                'global_1st': global_1st_k, 'global_1st_err': global_1st_err,
                'global_2nd': global_2nd_k, 'global_2nd_err': global_2nd_err,
            })
    
    # Print table
    print(f"\n{'Target':<20} {'Metric':<8} {'Frames':<8} | {'Mode Best':<18} {'Mode 2nd':<18} | {'Global Best':<20} {'Global 2nd':<20}")
    print("-" * 120)
    
    for row in table_data:
        if row['frames'] == 0:
            print(f"{row['target']:<20} {row['metric']:<8} {'N/A':<8} | {'-':<18} {'-':<18} | {'-':<20} {'-':<20}")
        else:
            mode_1st_str = f"k={row['mode_1st']} ({row['mode_1st_pct']:.1f}%)"
            mode_2nd_str = f"k={row['mode_2nd']} ({row['mode_2nd_pct']:.1f}%)" if row['mode_2nd'] != '-' else "-"
            global_1st_str = f"k={row['global_1st']} (err={row['global_1st_err']:.4f})" if row['global_1st'] != '-' else "-"
            global_2nd_str = f"k={row['global_2nd']} (err={row['global_2nd_err']:.4f})" if row['global_2nd'] != '-' else "-"
            
            print(f"{row['target']:<20} {row['metric']:<8} {row['frames']:<8} | {mode_1st_str:<18} {mode_2nd_str:<18} | {global_1st_str:<20} {global_2nd_str:<20}")
    
    print("-" * 120)
    print("\nMode Best/2nd: Per-frame analysis - which k is most/2nd-most frequent")
    print("Global Best/2nd: Global analysis - which k has lowest/2nd-lowest mean error across all frames")


def save_results(results: dict, output_path: Path, targets: list):
    """Save detailed results to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    
    for target_name in targets:
        target_results = results[target_name]
        serializable_results[target_name] = {}
        
        for metric, data in target_results.items():
            if len(data['best_k']) > 0:
                mode_k, mode_count = get_mode(np.array(data['best_k']))
                mode_pct = 100 * mode_count / len(data['best_k'])
            else:
                mode_k, mode_count, mode_pct = None, None, None
            
            # Compute global k analysis
            distances_per_k = data.get('distances_per_k', {})
            global_k_analysis = []
            for k, dists in distances_per_k.items():
                if len(dists) > 0:
                    global_k_analysis.append({
                        'k': k,
                        'mean_error': float(np.mean(dists)),
                        'total_error': float(np.sum(dists)),
                        'num_frames': len(dists)
                    })
            global_k_analysis_sorted = sorted(global_k_analysis, key=lambda x: x['mean_error'])
            
            best_global_k = global_k_analysis_sorted[0]['k'] if global_k_analysis_sorted else None
            best_global_mean_error = global_k_analysis_sorted[0]['mean_error'] if global_k_analysis_sorted else None
            
            serializable_results[target_name][metric] = {
                'best_k': [int(k) for k in data['best_k']],
                'best_distance': [float(d) for d in data['best_distance']],
                'per_episode': data['per_episode'],
                'summary': {
                    'mode_k': int(mode_k) if mode_k is not None else None,
                    'mode_count': int(mode_count) if mode_count is not None else None,
                    'mode_percentage': float(mode_pct) if mode_pct is not None else None,
                    'median_k': float(np.median(data['best_k'])) if len(data['best_k']) > 0 else None,
                    'mean_k': float(np.mean(data['best_k'])) if len(data['best_k']) > 0 else None,
                    'std_k': float(np.std(data['best_k'])) if len(data['best_k']) > 0 else None,
                    'mean_distance': float(np.mean(data['best_distance'])) if len(data['best_distance']) > 0 else None,
                    'best_global_k': best_global_k,
                    'best_global_mean_error': best_global_mean_error,
                },
                'global_k_analysis': global_k_analysis_sorted[:15]  # Top 15
            }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze action-to-state tracking delay')
    parser.add_argument('--data_dir', type=str, default='.', 
                        help='Directory containing episode folders')
    parser.add_argument('--max_k', type=int, default=30,
                        help='Maximum lookahead steps to consider (default: 30)')
    parser.add_argument('--metrics', type=str, nargs='+', default=['l2', 'l1'],
                        choices=['l1', 'l2'],
                        help='Distance metrics to use (default: l2 l1)')
    parser.add_argument('--targets', type=str, nargs='+', default=['body', 'hand_left', 'hand_right'],
                        choices=['body', 'hand_left', 'hand_right'],
                        help='Targets to analyze (default: body hand_left hand_right)')
    parser.add_argument('--output', type=str, default='action_delay_analysis.json',
                        help='Output JSON file for detailed results')
    parser.add_argument('--single_episode', type=str, default=None,
                        help='Analyze a single episode (provide episode folder name)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.data_dir)
    targets = args.targets
    
    if args.single_episode:
        # Analyze single episode
        ep_path = base_dir / args.single_episode / 'data.json'
        if not ep_path.exists():
            print(f"Error: {ep_path} not found")
            return
        
        print(f"Analyzing single episode: {args.single_episode}")
        print(f"Targets: {targets}")
        results = analyze_episode(ep_path, args.max_k, args.metrics, targets)
        
        # Wrap results to match expected format
        wrapped_results = {}
        for target_name in targets:
            wrapped_results[target_name] = {metric: {
                'best_k': results[target_name][metric]['best_k'],
                'best_distance': results[target_name][metric]['best_distance'],
                'per_episode': {args.single_episode: {
                    'mean_k': np.mean(results[target_name][metric]['best_k']) if results[target_name][metric]['best_k'] else 0,
                    'median_k': np.median(results[target_name][metric]['best_k']) if results[target_name][metric]['best_k'] else 0,
                    'std_k': np.std(results[target_name][metric]['best_k']) if results[target_name][metric]['best_k'] else 0,
                    'mean_dist': np.mean(results[target_name][metric]['best_distance']) if results[target_name][metric]['best_distance'] else 0,
                    'num_frames': len(results[target_name][metric]['best_k'])
                }}
            } for metric in args.metrics}
        
        print_summary(wrapped_results, args.metrics, targets)
        save_results(wrapped_results, base_dir / args.output, targets)
    else:
        # Analyze all episodes
        print(f"Analyzing all episodes in: {base_dir}")
        print(f"Max lookahead (k): {args.max_k}")
        print(f"Metrics: {args.metrics}")
        print(f"Targets: {targets}")
        print()
        
        results = analyze_all_episodes(base_dir, args.max_k, args.metrics, targets)
        print_summary(results, args.metrics, targets)
        save_results(results, base_dir / args.output, targets)


if __name__ == '__main__':
    main()
