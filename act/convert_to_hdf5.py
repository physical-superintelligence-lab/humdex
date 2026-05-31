"""
Convert JSON + JPEG dataset to compressed HDF5 format.

Converts episodic dataset from:
  episode_XXXX/data.json + episode_XXXX/rgb/*.jpg

To a single compressed HDF5 file with structure:
  /episode_0000/state_body                     (T, 31)  roll/pitch + 29 joints
  /episode_0000/state_wuji_hand_left           (T, 20)
  /episode_0000/state_wuji_hand_right          (T, 20)
  /episode_0000/action_body                    (T, 35)
  /episode_0000/action_wuji_qpos_target_left   (T, 20)
  /episode_0000/action_wuji_qpos_target_right  (T, 20)
  /episode_0000/head                           (T,) JPEG bytes

If raw data has 34D state_body (with angular velocity prefix), it is
automatically trimmed to 31D by dropping the first 3 dimensions.

Usage:
  # Single dataset directory
  python convert_to_hdf5.py --dataset_dir /path/to/data --output dataset.hdf5

  # Multiple dataset directories (merged + re-indexed)
  python convert_to_hdf5.py --dataset_dirs /path/to/d1 /path/to/d2 --output merged.hdf5
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def convert_episode_to_hdf5(
    episode_dir: str,
    hdf5_group,
    source_root: Optional[str] = None,
    source_episode_name: Optional[str] = None,
):
    """
    Convert a single episode from JSON+JPEG to HDF5 format.

    Args:
        episode_dir: Path to episode_XXXX directory
        hdf5_group: HDF5 group to write data to
        source_root: Optional source root folder (for merged datasets)
        source_episode_name: Optional original episode folder name (for merged datasets)
    """
    # Load JSON data
    json_path = os.path.join(episode_dir, 'data.json')
    with open(json_path, 'r') as f:
        episode_data = json.load(f)

    timesteps = episode_data['data']
    num_timesteps = len(timesteps)

    # Token-level (sonic) episodes store a 64-D encoder token as the body action
    # (action_token) instead of a 35-D joint action_body. Detect and branch on it.
    is_token = any(isinstance(ts, dict) and ts.get('action_token') is not None for ts in timesteps)

    # Initialize arrays for each data type
    state_body_list = []
    state_wuji_hand_left_list = []
    state_wuji_hand_right_list = []
    action_body_list = []
    action_token_list = []
    action_wuji_qpos_target_left_list = []
    action_wuji_qpos_target_right_list = []
    rgb_images = []
    missing_right_count = 0

    # Collect all data from timesteps
    for ts in timesteps:
        # States
        raw_state_body = np.array(ts['state_body'], dtype=np.float32)
        if len(raw_state_body) >= 34:
            # Extract 31D: skip ang_vel[0:3], keep roll/pitch[3:5] + joints[5:34]
            state_body_list.append(raw_state_body[3:34])
        else:
            state_body_list.append(raw_state_body)
        hand_left = ts.get('state_wuji_hand_left')
        state_wuji_hand_left_list.append(
            np.array(hand_left, dtype=np.float32) if hand_left is not None
            else np.zeros((20,), dtype=np.float32))
        if 'state_wuji_hand_right' in ts:
            state_wuji_hand_right_list.append(np.array(ts['state_wuji_hand_right'], dtype=np.float32))
        else:
            state_wuji_hand_right_list.append(np.zeros((20,), dtype=np.float32))
            missing_right_count += 1

        # Actions: token-level stores a 64-D encoder token; joint stores 35-D action_body
        if is_token:
            action_token_list.append(np.array(ts['action_token'], dtype=np.float32))
        else:
            action_body_list.append(np.array(ts['action_body'], dtype=np.float32))
        action_wuji_qpos_target_left_list.append(np.array(ts['action_wuji_qpos_target_left'], dtype=np.float32))
        if 'action_wuji_qpos_target_right' in ts:
            action_wuji_qpos_target_right_list.append(np.array(ts['action_wuji_qpos_target_right'], dtype=np.float32))
        else:
            action_wuji_qpos_target_right_list.append(np.zeros((20,), dtype=np.float32))
            missing_right_count += 1

        # Load RGB image as JPEG bytes
        image_path = os.path.join(episode_dir, ts['rgb'])
        with open(image_path, 'rb') as f:
            jpeg_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        rgb_images.append(jpeg_bytes)

    # Convert lists to numpy arrays
    state_body = np.array(state_body_list, dtype=np.float32)
    state_wuji_hand_left = np.array(state_wuji_hand_left_list, dtype=np.float32)
    state_wuji_hand_right = np.array(state_wuji_hand_right_list, dtype=np.float32)
    action_body = np.array(action_body_list, dtype=np.float32)
    action_token = np.array(action_token_list, dtype=np.float32)
    action_wuji_qpos_target_left = np.array(action_wuji_qpos_target_left_list, dtype=np.float32)
    action_wuji_qpos_target_right = np.array(action_wuji_qpos_target_right_list, dtype=np.float32)
    rgb = np.empty(len(rgb_images), dtype=object)
    for i, img_bytes in enumerate(rgb_images):
        rgb[i] = img_bytes

    compression = 'gzip'
    compression_opts = 4

    hdf5_group.create_dataset('state_body', data=state_body,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('state_wuji_hand_left', data=state_wuji_hand_left,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('state_wuji_hand_right', data=state_wuji_hand_right,
                             compression=compression, compression_opts=compression_opts)
    if is_token:
        hdf5_group.create_dataset('action_token', data=action_token,
                                 compression=compression, compression_opts=compression_opts)
    else:
        hdf5_group.create_dataset('action_body', data=action_body,
                                 compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('action_wuji_qpos_target_left', data=action_wuji_qpos_target_left,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('action_wuji_qpos_target_right', data=action_wuji_qpos_target_right,
                             compression=compression, compression_opts=compression_opts)

    dt = h5py.vlen_dtype(np.dtype('uint8'))
    hdf5_group.create_dataset('head', data=rgb, dtype=dt)

    hdf5_group.attrs['num_timesteps'] = num_timesteps
    hdf5_group.attrs['body_action_repr'] = 'token' if is_token else 'joint'
    hdf5_group.attrs['episode_dir'] = os.path.basename(episode_dir)
    hdf5_group.attrs['has_right_hand'] = (missing_right_count == 0)
    hdf5_group.attrs['missing_right_hand_count'] = int(missing_right_count)
    if source_root is not None:
        hdf5_group.attrs['source_root'] = str(source_root)
    if source_episode_name is not None:
        hdf5_group.attrs['source_episode'] = str(source_episode_name)
    if 'info' in episode_data:
        hdf5_group.attrs['info'] = json.dumps(episode_data['info'])
    if 'text' in episode_data:
        hdf5_group.attrs['text'] = json.dumps(episode_data['text'])

    return num_timesteps


def _list_episode_dirs(dataset_dir: str) -> List[str]:
    """List episode_XXXX directories under a dataset_dir (sorted)."""
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    episode_dirs = sorted([
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if d.startswith('episode_') and os.path.isdir(os.path.join(dataset_dir, d))
    ])
    return episode_dirs


def _collect_episode_dirs(dataset_dirs: List[str], num_episodes: Optional[int]) -> List[Tuple[str, str, str]]:
    """Collect episodes across multiple roots.

    Returns list of tuples: (source_root, episode_name, episode_dir_abs)
    """
    all_eps: List[Tuple[str, str, str]] = []
    for root in dataset_dirs:
        root = os.path.abspath(root)
        for ep_dir in _list_episode_dirs(root):
            ep_name = os.path.basename(ep_dir)
            all_eps.append((root, ep_name, ep_dir))

    # Deterministic ordering: by source_root then episode_name
    all_eps.sort(key=lambda x: (x[0], x[1]))

    if num_episodes is not None:
        all_eps = all_eps[:num_episodes]
    return all_eps


def convert_dataset_to_hdf5(dataset_dirs, output_path, num_episodes=None):
    """
    Convert entire dataset from JSON+JPEG to single HDF5 file.

    Args:
        dataset_dirs: Directory (str) OR list of directories containing episode_XXXX folders
        output_path: Path to output HDF5 file
        num_episodes: Number of episodes to convert (None = all episodes)
    """
    if isinstance(dataset_dirs, str):
        dataset_dirs = [dataset_dirs]
    dataset_dirs = [os.path.abspath(d) for d in dataset_dirs]

    episodes = _collect_episode_dirs(dataset_dirs, num_episodes=num_episodes)

    print(f"Found {len(episodes)} episodes to convert")
    print(f"Output: {output_path}")
    print(f"state_body: 31D (roll/pitch + joints)")
    print()

    with h5py.File(output_path, 'w') as hdf5_file:
        hdf5_file.attrs['num_episodes'] = len(episodes)
        hdf5_file.attrs['dataset_dirs'] = json.dumps(dataset_dirs)
        hdf5_file.attrs['state_body_dim'] = 31

        total_timesteps = 0
        episode_lengths = []
        episode_source_map: List[Dict[str, Any]] = []

        # Convert each episode
        for new_idx, (src_root, src_ep_name, episode_dir) in enumerate(tqdm(episodes, desc="Converting episodes")):
            # Re-index episodes sequentially to avoid collisions across multiple roots
            episode_name = f'episode_{new_idx:04d}'

            # Create group for this episode
            episode_group = hdf5_file.create_group(episode_name)

            num_timesteps = convert_episode_to_hdf5(
                episode_dir,
                episode_group,
                source_root=src_root,
                source_episode_name=src_ep_name,
            )

            total_timesteps += num_timesteps
            episode_lengths.append(num_timesteps)
            episode_source_map.append({
                "episode_new": episode_name,
                "source_root": src_root,
                "source_episode": src_ep_name,
            })

        # Save summary statistics
        hdf5_file.attrs['total_timesteps'] = total_timesteps
        hdf5_file.attrs['min_episode_length'] = min(episode_lengths)
        hdf5_file.attrs['max_episode_length'] = max(episode_lengths)
        hdf5_file.attrs['mean_episode_length'] = np.mean(episode_lengths)
        hdf5_file.attrs['episode_source_map'] = json.dumps(episode_source_map)

    # Print statistics
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)

    # Get file sizes
    original_size = get_directory_size([ep[2] for ep in episodes])
    hdf5_size = os.path.getsize(output_path)

    print(f"\nOriginal size: {original_size / 1024**2:.2f} MB")
    print(f"HDF5 size:     {hdf5_size / 1024**2:.2f} MB")
    print(f"Compression ratio: {original_size / hdf5_size:.2f}x")
    print(f"Space saved: {(original_size - hdf5_size) / 1024**2:.2f} MB ({100 * (1 - hdf5_size/original_size):.1f}%)")

    print(f"\nDataset statistics:")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Episode length: {min(episode_lengths)} - {max(episode_lengths)} (mean: {np.mean(episode_lengths):.1f})")
    print(f"  state_body dim: 31D")


def get_directory_size(episode_dirs: List[str]):
    """Calculate total size of episode directories (sum)."""
    total_size = 0
    for episode_dir in episode_dirs:
        for dirpath, dirnames, filenames in os.walk(episode_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    return total_size


def verify_conversion(dataset_dir, hdf5_path, episode_idx=0):
    """Verify conversion by comparing original JSON data with HDF5 contents."""
    from io import BytesIO
    from PIL import Image
    import random

    print(f"\nVerifying episode {episode_idx}...")

    episode_dir = os.path.join(dataset_dir, f'episode_{episode_idx:04d}')
    json_path = os.path.join(episode_dir, 'data.json')
    with open(json_path, 'r') as f:
        original_data = json.load(f)

    with h5py.File(hdf5_path, 'r') as f:
        hdf5_episode = f[f'episode_{episode_idx:04d}']

        num_timesteps_original = len(original_data['data'])
        num_timesteps_hdf5 = hdf5_episode.attrs['num_timesteps']

        print(f"  Timesteps: {num_timesteps_original} (original) vs {num_timesteps_hdf5} (HDF5)")
        assert num_timesteps_original == num_timesteps_hdf5, "Timestep count mismatch!"

        test_indices = random.sample(range(num_timesteps_original), min(3, num_timesteps_original))

        for ts_idx in test_indices:
            ts_original = original_data['data'][ts_idx]

            state_body_original = np.array(ts_original['state_body'], dtype=np.float32)
            if len(state_body_original) >= 34:
                state_body_original = state_body_original[3:34]
            assert np.allclose(state_body_original, hdf5_episode['state_body'][ts_idx])

            assert np.allclose(
                np.array(ts_original['action_body'], dtype=np.float32),
                hdf5_episode['action_body'][ts_idx],
            ), f"action_body mismatch at timestep {ts_idx}"

            for key in ['state_wuji_hand_left', 'state_wuji_hand_right',
                        'action_wuji_qpos_target_left', 'action_wuji_qpos_target_right']:
                val = ts_original.get(key)
                original = np.array(val, dtype=np.float32) if val is not None else np.zeros((20,), dtype=np.float32)
                assert np.allclose(original, hdf5_episode[key][ts_idx]), f"{key} mismatch at timestep {ts_idx}"

            image_path = os.path.join(episode_dir, ts_original['rgb'])
            image_original = np.array(Image.open(image_path), dtype=np.uint8)
            jpeg_bytes = hdf5_episode['head'][ts_idx]
            image_hdf5 = np.array(Image.open(BytesIO(jpeg_bytes)), dtype=np.uint8)
            max_diff = np.max(np.abs(image_original.astype(np.int16) - image_hdf5.astype(np.int16)))
            assert max_diff <= 5, f"Head image differs too much at timestep {ts_idx} (max diff: {max_diff})"

        print(f"  Verified {len(test_indices)} random timesteps - all match!")

    print("  Verification passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON+JPEG dataset to HDF5 format')
    parser.add_argument('--dataset_dir', type=str, default=None,
                       help='Directory containing episode_XXXX folders (single source)')
    parser.add_argument('--dataset_dirs', type=str, nargs='+', default=None,
                       help='One or more directories, each containing episode_XXXX folders (will be merged + re-indexed)')
    parser.add_argument('--output', type=str, default='dataset.hdf5',
                       help='Output HDF5 file path')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of episodes to convert (default: all)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify conversion after completion')

    args = parser.parse_args()

    if args.dataset_dirs is not None and len(args.dataset_dirs) > 0:
        dataset_dirs = args.dataset_dirs
    elif args.dataset_dir is not None:
        dataset_dirs = args.dataset_dir
    else:
        raise ValueError("Please pass --dataset_dir or --dataset_dirs")

    convert_dataset_to_hdf5(
        dataset_dirs=dataset_dirs,
        output_path=args.output,
        num_episodes=args.num_episodes,
    )

    if args.verify:
        first_dir = args.dataset_dir or (args.dataset_dirs[0] if args.dataset_dirs else None)
        if first_dir is not None:
            verify_conversion(first_dir, args.output, episode_idx=0)
        print("\nAll verification checks passed!")
