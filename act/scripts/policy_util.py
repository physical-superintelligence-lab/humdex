"""Policy inference wrapper and HDF5 dataset reader."""

import os
import pickle
from typing import Optional

import numpy as np
import h5py


# =============================================================================
# ACT Policy Wrapper
# =============================================================================

class ACTPolicyWrapper:
    """Wrapper for ACT policy inference."""

    def __init__(self, ckpt_dir: str, config, policy_config: Optional[dict] = None):
        """
        Args:
            config: Object with attributes: use_gpu, chunk_size, temporal_agg, action_dim.
        """
        self.config = config
        self.device = 'cuda' if config.use_gpu else 'cpu'

        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        print(f"Loaded normalization stats from {stats_path}")
        print(f"  qpos_mean shape: {self.stats['qpos_mean'].shape}")
        print(f"  action_mean shape: {self.stats['action_mean'].shape}")

        self.policy = self._load_policy(ckpt_dir, policy_config)
        self.policy.eval()

        self.temporal_agg = config.temporal_agg
        self.chunk_size = config.chunk_size
        self.all_time_actions = None
        self.t = 0

    def _load_policy(self, ckpt_dir: str, policy_config: Optional[dict] = None):
        """Load ACT policy from checkpoint."""
        import torch
        from policy import ACTPolicy

        if policy_config is None:
            policy_config = {
                'lr': 1e-5,
                'lr_backbone': 1e-5,
                'num_queries': self.config.chunk_size,
                'kl_weight': 10,
                'hidden_dim': 512,
                'dim_feedforward': 3200,
                'backbone': 'resnet18',
                'enc_layers': 4,
                'dec_layers': 7,
                'nheads': 8,
                'camera_names': ['head'],
            }

        policy_config['state_dim'] = self.stats['state_dim']
        policy_config['action_dim'] = self.stats['action_dim']

        print(f"Policy config: hidden_dim={policy_config['hidden_dim']}, "
              f"chunk_size={policy_config['num_queries']}, "
              f"state_dim={policy_config['state_dim']}, "
              f"action_dim={policy_config['action_dim']}")

        policy = ACTPolicy(policy_config)

        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')

        state_dict = torch.load(ckpt_path, map_location=self.device)
        loading_status = policy.load_state_dict(state_dict)
        print(f"Loading status: {loading_status}")

        policy.to(self.device)

        print(f"Loaded policy from {ckpt_path}")
        return policy

    def reset(self):
        """Reset temporal aggregation state."""
        self.t = 0
        self.all_time_actions = None

    def preprocess_qpos(self, qpos: np.ndarray):
        """Normalize qpos."""
        import torch
        qpos_norm = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0)
        if self.config.use_gpu:
            qpos_tensor = qpos_tensor.cuda()
        return qpos_tensor

    def preprocess_image(self, image: np.ndarray):
        """Normalize image to (1, 1, C, H, W) tensor."""
        import torch
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, np.newaxis, ...]
        image_tensor = torch.from_numpy(image).float()
        if self.config.use_gpu:
            image_tensor = image_tensor.cuda()
        return image_tensor

    def postprocess_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action."""
        return action * self.stats['action_std'] + self.stats['action_mean']

    def __call__(self, qpos: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Run policy inference.

        Args:
            qpos: (state_dim,) state - 51 (single hand) or 71 (both hands)
            image: (H, W, 3) RGB image

        Returns:
            action: (action_dim,) denormalized action
                    55D for single hand, 75D for both hands
        """
        import torch

        qpos_tensor = self.preprocess_qpos(qpos)
        image_tensor = self.preprocess_image(image)

        with torch.inference_mode():
            if self.temporal_agg:
                return self._infer_with_temporal_agg(qpos_tensor, image_tensor)
            else:
                return self._infer_simple(qpos_tensor, image_tensor)

    def _infer_simple(self, qpos_tensor, image_tensor) -> np.ndarray:
        """Simple inference without temporal aggregation."""
        import torch

        if self.t % self.config.chunk_size == 0:
            self._cached_actions = self.policy(qpos_tensor, image_tensor)

        action_idx = self.t % self.config.chunk_size
        raw_action = self._cached_actions[0, action_idx].cpu().numpy()
        self.t += 1

        return self.postprocess_action(raw_action)

    def _infer_with_temporal_agg(self, qpos_tensor, image_tensor) -> np.ndarray:
        """Inference with temporal aggregation (exponential weighting)."""
        import torch

        max_timesteps = 5000

        if self.all_time_actions is None:
            self.all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + self.chunk_size, self.config.action_dim]
            )
            if self.config.use_gpu:
                self.all_time_actions = self.all_time_actions.cuda()

        t = self.t

        if t >= max_timesteps:
            if not hasattr(self, "_temporal_agg_clamp_warned") or not self._temporal_agg_clamp_warned:
                print(
                    f"[TemporalAgg] t={t} exceeds buffer size {max_timesteps}. "
                    f"Clamping to {max_timesteps-1}. Consider increasing max_timesteps "
                    f"if your episodes are this long."
                )
                self._temporal_agg_clamp_warned = True
            t = max_timesteps - 1

        all_actions = self.policy(qpos_tensor, image_tensor)

        all_actions_np = all_actions[0].cpu().numpy()
        all_actions_denorm = self.postprocess_action(all_actions_np)

        all_actions_abs_tensor = torch.from_numpy(all_actions_denorm).float()
        if self.config.use_gpu:
            all_actions_abs_tensor = all_actions_abs_tensor.cuda()

        self.all_time_actions[t, t:t+self.chunk_size] = all_actions_abs_tensor

        actions_for_curr_step = self.all_time_actions[:, t]
        actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]

        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).float().unsqueeze(1)
        if self.config.use_gpu:
            exp_weights = exp_weights.cuda()

        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0).cpu().numpy()
        self.t += 1

        return raw_action


# =============================================================================
# HDF5 Dataset Reader
# =============================================================================

class DatasetReader:
    """Read episodes from HDF5 dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.episode_ids = self._get_episode_ids()
        print(f"Dataset: {dataset_path}")
        print(f"Episodes: {len(self.episode_ids)} ({self.episode_ids[:5]}...)")

    def _get_episode_ids(self) -> list:
        with h5py.File(self.dataset_path, 'r') as f:
            ids = [int(k.split('_')[1]) for k in f.keys() if k.startswith('episode_')]
        return sorted(ids)

    def load_episode(self, episode_id: int, load_observations: bool = False, hand_side: str = "left") -> dict:
        """
        Load a single episode.

        Args:
            episode_id: Episode index
            load_observations: If True, also load qpos and images for policy input
            hand_side: "left", "right", or "both" to select which hand data to load

        Returns:
            dict with keys:
                - action_body: (T, 35) GT body actions
                - action_hand: (T, 20 or 40) GT Wuji hand target(s)
                - num_timesteps: int
                - text: str
                - qpos: (T, state_dim) state observations (if load_observations=True)
                - images: (T, H, W, 3) RGB images (if load_observations=True)
        """
        if hand_side not in ["left", "right", "both"]:
            raise ValueError(f"hand_side must be 'left', 'right', or 'both', got {hand_side}")

        with h5py.File(self.dataset_path, 'r') as f:
            key = f'episode_{episode_id:04d}'
            if key not in f:
                raise ValueError(f"Episode {episode_id} not found")

            ep = f[key]
            data = {
                'action_body': ep['action_body'][:],
                'num_timesteps': ep.attrs['num_timesteps'],
                'text': ep.attrs.get('text', '{}'),
            }

            if hand_side == "both":
                hand_left_key = 'action_wuji_qpos_target_left'
                hand_right_key = 'action_wuji_qpos_target_right'

                T = int(ep.attrs['num_timesteps'])
                if hand_left_key in ep:
                    action_hand_left = ep[hand_left_key][:]
                else:
                    action_hand_left = np.zeros((T, 20), dtype=np.float32)

                if hand_right_key in ep:
                    action_hand_right = ep[hand_right_key][:]
                else:
                    action_hand_right = np.zeros((T, 20), dtype=np.float32)

                data['action_hand'] = np.concatenate([action_hand_left, action_hand_right], axis=1)
            else:
                hand_action_key = f'action_wuji_qpos_target_{hand_side}'
                if hand_action_key in ep:
                    data['action_hand'] = ep[hand_action_key][:]
                else:
                    T = int(ep.attrs['num_timesteps'])
                    data['action_hand'] = np.zeros((T, 20), dtype=np.float32)

            if load_observations:
                state_body = ep['state_body'][:]

                if hand_side == "both":
                    hand_left_key = 'state_wuji_hand_left'
                    hand_right_key = 'state_wuji_hand_right'

                    T = int(ep.attrs['num_timesteps'])
                    if hand_left_key in ep:
                        state_hand_left = ep[hand_left_key][:]
                    else:
                        state_hand_left = np.zeros((T, 20), dtype=np.float32)

                    if hand_right_key in ep:
                        state_hand_right = ep[hand_right_key][:]
                    else:
                        state_hand_right = np.zeros((T, 20), dtype=np.float32)

                    data['qpos'] = np.concatenate([state_body, state_hand_left, state_hand_right], axis=1)
                else:
                    hand_state_key = f'state_wuji_hand_{hand_side}'
                    if hand_state_key in ep:
                        state_hand = ep[hand_state_key][:]
                    else:
                        T = int(ep.attrs['num_timesteps'])
                        state_hand = np.zeros((T, 20), dtype=np.float32)
                    data['qpos'] = np.concatenate([state_body, state_hand], axis=1)

                head = ep['head'][:]
                if isinstance(head, np.ndarray) and (head.dtype == object or head.ndim == 1):
                    try:
                        import cv2
                        decoded = []
                        fallback_hw = (480, 640)
                        for i in range(len(head)):
                            buf = head[i]
                            if isinstance(buf, (bytes, bytearray)):
                                arr = np.frombuffer(buf, dtype=np.uint8)
                            else:
                                arr = np.asarray(buf, dtype=np.uint8).reshape(-1)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if bgr is None:
                                h, w = fallback_hw
                                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                            else:
                                rgb = bgr[:, :, ::-1]
                                fallback_hw = (rgb.shape[0], rgb.shape[1])
                            decoded.append(rgb)
                        data['images'] = np.stack(decoded, axis=0)
                    except Exception as e:
                        print(f"[DatasetReader] Warning: failed to decode images from 'head': {e}")
                        data['images'] = head
                else:
                    data['images'] = head

        return data

    def random_episode_id(self) -> int:
        return np.random.choice(self.episode_ids)
