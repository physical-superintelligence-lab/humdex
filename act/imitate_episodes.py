import os

# Force non-GUI matplotlib backend when running headless MuJoCo to avoid Tk crashes.
_MUJOCO_GL = str(os.environ.get("MUJOCO_GL", "") or "").strip().lower()
if _MUJOCO_GL in ("egl", "osmesa"):
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass

import gc
import glob
import re
import time
import pickle
import argparse
from copy import deepcopy

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataset import load_data, get_num_episodes, get_episode_ids, get_norm_stats
from dataset import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy


def main(args):
    set_seed(1)

    ckpt_root = args['ckpt_root']
    policy_class = args['policy_class']
    task_name = args['task_name']
    resume = bool(args.get('resume', False))
    resume_ckpt = args.get('resume_ckpt', None)
    resume_save_every = int(args.get('resume_save_every', 100))
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    dataset_path = args['dataset_path']
    use_rgb = not args['no_rgb']
    hand_side = args.get('hand_side', 'left')
    channel = args.get('channel', 'twist2')
    sequential_training = args.get('sequential_training', False)
    epochs_per_dataset = args.get('epochs_per_dataset', None)
    ckpt_prefix = str(args.get('ckpt_prefix', '') or '').strip()

    if isinstance(dataset_path, str):
        dataset_paths = [dataset_path]
    else:
        dataset_paths = list(dataset_path)

    # Validate sequential training arguments
    if sequential_training:
        if epochs_per_dataset is None:
            raise ValueError("--sequential_training requires --epochs_per_dataset")

        # Two modes:
        # 1. len(epochs_per_dataset) == len(dataset_paths): train each dataset separately
        # 2. len(epochs_per_dataset) == 2: stage1 = mix all but last, stage2 = last dataset only
        if len(epochs_per_dataset) == len(dataset_paths):
            print(f"\n=== Sequential Training Mode ===")
            print(f"Will train on {len(dataset_paths)} datasets sequentially:")
            for i, (path, epochs) in enumerate(zip(dataset_paths, epochs_per_dataset)):
                print(f"  Stage {i+1}: {os.path.basename(path)} for {epochs} epochs")
            print()
        elif len(epochs_per_dataset) == 2 and len(dataset_paths) >= 2:
            print(f"\n=== Sequential Training Mode (Two-Stage) ===")
            print(f"Stage 1: Mixed training on {len(dataset_paths)-1} dataset(s) for {epochs_per_dataset[0]} epochs")
            for path in dataset_paths[:-1]:
                print(f"  - {os.path.basename(path)}")
            print(f"Stage 2: Training on last dataset for {epochs_per_dataset[1]} epochs")
            print(f"  - {os.path.basename(dataset_paths[-1])}")
            print()
        else:
            raise ValueError(
                f"--epochs_per_dataset must have either {len(dataset_paths)} values (one per dataset) "
                f"or 2 values (stage1=mix of first {len(dataset_paths)-1}, stage2=last dataset)"
            )

    dataset_path_display = dataset_paths[0] if len(dataset_paths) == 1 else f"{len(dataset_paths)} files"

    if resume:
        ckpt_dir = args.get('ckpt_dir', None)
        if ckpt_dir is None:
            raise ValueError("Resume training requires --ckpt_dir.")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        folder = f"{ckpt_prefix}_{ts}" if ckpt_prefix else ts
        ckpt_dir = os.path.join(ckpt_root, task_name, folder)

    state_body_dim = 29 if channel == "sonic" else 31
    action_body_dim = 64 if channel == "sonic" else 35
    hand_dim = 20 if hand_side in ("left", "right") else 40
    state_dim = state_body_dim + hand_dim
    camera_names = ['head']
    print(f"channel: {channel} | state_dim: {state_dim} (body={state_body_dim} + hand={hand_dim}), action_body_dim: {action_body_dim}, hand_side: {hand_side}")

    # Auto-detect number of episodes
    num_episodes = get_num_episodes(dataset_paths)
    print(f"Found {num_episodes} episodes from {dataset_path_display}")

    # Build policy config
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': camera_names,
            'state_dim': state_dim,
            'action_dim': action_body_dim + hand_dim,
        }
        os.environ.setdefault("ACT_CHUNK_SIZE", str(int(args["chunk_size"])))
    elif policy_class == 'CNNMLP':
        policy_config = {
            'lr': args['lr'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'num_queries': 1,
            'camera_names': camera_names,
            'state_dim': state_dim,
            'action_dim': action_body_dim + hand_dim,
        }
    else:
        raise NotImplementedError(f"Unknown policy class: {policy_class}")

    config = {
        'ckpt_dir': ckpt_dir,
        'num_epochs': num_epochs,
        'state_dim': state_dim,
        'state_body_dim': state_body_dim,
        'action_body_dim': action_body_dim,
        'hand_dim': hand_dim,
        'total_action_dim': action_body_dim + hand_dim,
        'channel': channel,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'use_wandb': args['wandb'] and WANDB_AVAILABLE,
        'wandb_project': args['wandb_project'],
        'wandb_run_name': args['wandb_run_name'],
        'use_rgb': use_rgb,
        'hand_side': hand_side,
        'dataset_paths': dataset_paths,
        'resume': resume,
        'resume_ckpt': resume_ckpt,
        'resume_save_every': resume_save_every,
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if sequential_training:
        all_stage_results = []
        prev_checkpoint = None
        unified_norm_stats = None

        if args.get('sequential_unified_stats', False):
            print(f"\n{'='*80}")
            print("Computing Unified Normalization Stats on ALL Datasets")
            print(f"{'='*80}")
            all_episode_ids = get_episode_ids(dataset_paths)
            unified_norm_stats = get_norm_stats(
                dataset_paths, all_episode_ids,
                hand_side=hand_side,
            )
            unified_norm_stats['state_body_dim'] = state_body_dim
            unified_norm_stats['action_body_dim'] = action_body_dim
            unified_stats_path = os.path.join(ckpt_dir, 'dataset_stats_unified.pkl')
            with open(unified_stats_path, 'wb') as f:
                pickle.dump(unified_norm_stats, f)
            print(f"Saved unified stats to {unified_stats_path}")
            print(f"  - Computed from {len(all_episode_ids)} episodes across {len(dataset_paths)} datasets")
            print(f"  - All stages will use these unified stats\n")

        # Determine stage configuration
        if len(epochs_per_dataset) == 2 and len(dataset_paths) >= 2:
            stage_configs = [
                (dataset_paths[:-1], epochs_per_dataset[0], 1),
                ([dataset_paths[-1]], epochs_per_dataset[1], 2),
            ]
        else:
            stage_configs = [
                ([path], epochs, idx + 1)
                for idx, (path, epochs) in enumerate(zip(dataset_paths, epochs_per_dataset))
            ]

        for stage_datasets, stage_epochs, stage_idx in stage_configs:
            print(f"\n{'='*80}")
            if len(stage_datasets) == 1:
                print(f"STAGE {stage_idx}/{len(stage_configs)}: Training on {os.path.basename(stage_datasets[0])}")
            else:
                print(f"STAGE {stage_idx}/{len(stage_configs)}: Mixed training on {len(stage_datasets)} datasets")
                for p in stage_datasets:
                    print(f"  - {os.path.basename(p)}")
            print(f"Epochs: {stage_epochs}")
            print(f"{'='*80}\n")

            train_dataloader, val_dataloader, stage_stats, _ = load_data(
                stage_datasets, None, camera_names,
                batch_size_train, batch_size_val,
                use_rgb=use_rgb, hand_side=hand_side, channel=channel,
                split_save_path=os.path.join(ckpt_dir, f"train_val_split_stage{stage_idx}.json"),
                val_robot_only=bool(args.get("val_robot_only", False)),
            )

            if unified_norm_stats is not None:
                stats = unified_norm_stats
                print(f"[Stage {stage_idx}] Using unified normalization stats\n")
            else:
                stats = stage_stats
                stats['state_body_dim'] = state_body_dim
                stats['action_body_dim'] = action_body_dim
                stats_path = os.path.join(ckpt_dir, f'dataset_stats_stage{stage_idx}.pkl')
                with open(stats_path, 'wb') as f:
                    pickle.dump(stats, f)
                print(f"Saved stage {stage_idx} dataset stats to {stats_path}\n")

            stage_config = deepcopy(config)
            stage_config['num_epochs'] = stage_epochs
            stage_config['dataset_paths'] = stage_datasets

            best_ckpt_info = train_bc(
                train_dataloader, val_dataloader, stage_config,
                norm_stats=stats,
                prev_checkpoint=prev_checkpoint,
                stage_idx=stage_idx,
                total_stages=len(stage_configs),
            )
            best_epoch, min_val_loss, best_state_dict = best_ckpt_info

            stage_ckpt_path = os.path.join(ckpt_dir, f'policy_stage{stage_idx}_best.ckpt')
            torch.save(best_state_dict, stage_ckpt_path)
            print(f'\nStage {stage_idx} complete: val loss {min_val_loss:.6f} @ epoch {best_epoch}')
            print(f'Saved to {stage_ckpt_path}\n')

            prev_checkpoint = stage_ckpt_path
            all_stage_results.append({
                'stage': stage_idx,
                'datasets': [os.path.basename(p) for p in stage_datasets],
                'epochs': stage_epochs,
                'best_epoch': best_epoch,
                'min_val_loss': min_val_loss,
            })

            if stage_idx < len(stage_configs):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f"\n{'='*80}")
        print("Sequential Training Complete!")
        print(f"{'='*80}")
        for result in all_stage_results:
            datasets_str = ', '.join(result['datasets'])
            print(f"Stage {result['stage']}: {datasets_str}")
            print(f"  Epochs: {result['epochs']}, Best: {result['best_epoch']}, Val Loss: {result['min_val_loss']:.6f}")
        print(f"\nFinal model saved to: {ckpt_path}")

    else:
        train_dataloader, val_dataloader, stats, _ = load_data(
            dataset_paths, num_episodes, camera_names,
            batch_size_train, batch_size_val,
            use_rgb=use_rgb, hand_side=hand_side,
            split_save_path=os.path.join(ckpt_dir, "train_val_split.json"),
            val_robot_only=bool(args.get("val_robot_only", False)),
        )

        stats['state_body_dim'] = state_body_dim
        stats['action_body_dim'] = action_body_dim
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Saved dataset stats to {stats_path}")

        best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, norm_stats=stats)
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch {best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        return ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        return CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError(f"Unknown policy class: {policy_class}")


def forward_pass(data, policy, pred_action=False):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    )
    if not pred_action:
        return policy(qpos_data, image_data, action_data, is_pad)
    else:
        return policy(qpos_data, image_data)


def visualize_predictions(data, a_hat, norm_stats, body_viz, hand_viz, ckpt_dir,
                          epoch, split='val', num_samples=4, action_body_dim=35,
                          hand_side="left", hand_viz_right=None):
    """Visualize GT and predicted actions for multiple samples. Returns path to saved video."""
    import cv2

    image_data, qpos_data, action_data, is_pad = data
    num_samples = min(num_samples, a_hat.shape[0])
    chunk_size = a_hat.shape[1]

    def add_label(frame, label):
        frame = frame.copy()
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return frame

    per_sample = []
    for i in range(num_samples):
        pad_mask = is_pad[i, :chunk_size].cpu().numpy()
        valid_len = max(int((~pad_mask).sum()), 1)

        pred_actions = a_hat[i, :valid_len].cpu().numpy() * norm_stats['action_std'] + norm_stats['action_mean']
        gt_actions = action_data[i, :valid_len].cpu().numpy() * norm_stats['action_std'] + norm_stats['action_mean']

        body_pred = pred_actions[:, :action_body_dim]
        body_gt = gt_actions[:, :action_body_dim]

        rgb = (image_data[i, 0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        body_gt_frames = body_viz.visualize(body_gt, verbose=False)
        body_pred_frames = body_viz.visualize(body_pred, verbose=False)

        if hand_side == "both":
            hand_left_gt_frames = hand_viz.visualize(gt_actions[:, 35:55], verbose=False)
            hand_left_pred_frames = hand_viz.visualize(pred_actions[:, 35:55], verbose=False)
            viz_right = hand_viz_right if hand_viz_right is not None else hand_viz
            hand_right_gt_frames = viz_right.visualize(gt_actions[:, 55:75], verbose=False)
            hand_right_pred_frames = viz_right.visualize(pred_actions[:, 55:75], verbose=False)
        else:
            hand_left_gt_frames = hand_viz.visualize(gt_actions[:, action_body_dim:], verbose=False)
            hand_left_pred_frames = hand_viz.visualize(pred_actions[:, action_body_dim:], verbose=False)
            hand_right_gt_frames = None
            hand_right_pred_frames = None

        h = body_gt_frames[0].shape[0]
        rgb_resized = cv2.resize(rgb, (int(rgb.shape[1] * h / rgb.shape[0]), h))

        sample_data = {
            "rgb": rgb_resized,
            "body_gt": body_gt_frames, "body_pred": body_pred_frames,
            "hand_left_gt": hand_left_gt_frames, "hand_left_pred": hand_left_pred_frames,
        }
        if hand_side == "both":
            sample_data["hand_right_gt"] = hand_right_gt_frames
            sample_data["hand_right_pred"] = hand_right_pred_frames

        per_sample.append(sample_data)

    video_path = os.path.join(ckpt_dir, f'{split}_viz_epoch_{epoch}.mp4')
    try:
        import imageio.v2 as imageio
        max_frames = max(max(len(s["body_gt"]) for s in per_sample), 1)
        writer = imageio.get_writer(video_path, fps=20, macro_block_size=1)
        try:
            for t in range(max_frames):
                rows = []
                for s in per_sample:
                    bgt = add_label(s["body_gt"][min(t, len(s["body_gt"]) - 1)], "Body GT")
                    bpred = add_label(s["body_pred"][min(t, len(s["body_pred"]) - 1)], "Body Pred")
                    hlgt = add_label(s["hand_left_gt"][min(t, len(s["hand_left_gt"]) - 1)], "L-Hand GT")
                    hlpred = add_label(s["hand_left_pred"][min(t, len(s["hand_left_pred"]) - 1)], "L-Hand Pred")

                    if hand_side == "both":
                        hrgt = add_label(s["hand_right_gt"][min(t, len(s["hand_right_gt"]) - 1)], "R-Hand GT")
                        hrpred = add_label(s["hand_right_pred"][min(t, len(s["hand_right_pred"]) - 1)], "R-Hand Pred")
                        combined = np.concatenate([s["rgb"], bgt, bpred, hlgt, hlpred, hrgt, hrpred], axis=1)
                    else:
                        combined = np.concatenate([s["rgb"], bgt, bpred, hlgt, hlpred], axis=1)

                    combined = np.ascontiguousarray(np.clip(combined, 0, 255).astype(np.uint8))
                    rows.append(combined)
                grid = np.ascontiguousarray(np.concatenate(rows, axis=0))
                writer.append_data(grid)
        finally:
            writer.close()
        return video_path
    except Exception as e:
        print(f"Visualization failed ({split}, epoch={epoch}): {e}")
        return None


def train_bc(train_dataloader, val_dataloader, config, norm_stats=None,
             prev_checkpoint=None, stage_idx=None, total_stages=None):
    """
    Main training loop. Supports both regular and sequential training.

    Args:
        prev_checkpoint: If set, load policy weights from this path before training
                        (used by sequential training to chain stages).
        stage_idx: If set, prefix checkpoint filenames and wandb with stage info.
        total_stages: Total number of stages (for logging only).
    """
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_wandb = config.get('use_wandb', False)
    action_body_dim = config.get('action_body_dim', 35)
    hand_side = config.get('hand_side', 'left')
    channel = config.get('channel', 'twist2')
    resume = bool(config.get('resume', False))
    resume_ckpt = config.get('resume_ckpt', None)
    resume_save_every = int(config.get('resume_save_every', 20))

    is_stage = stage_idx is not None
    stage_suffix = f"_stage{stage_idx}" if is_stage else ""
    stage_label = f"[Stage {stage_idx}/{total_stages}] " if is_stage else ""

    set_seed(seed)

    # Initialize wandb
    if use_wandb:
        run_name = config.get('wandb_run_name') or f"{config['task_name']}_seed{seed}_{time.strftime('%m%d_%H%M')}"
        if is_stage:
            run_name = f"{run_name}_stage{stage_idx}"
        wandb_config = {
            'task_name': config['task_name'],
            'policy_class': policy_class,
            'num_epochs': num_epochs,
            'seed': seed,
            'lr': config['lr'],
            'use_rgb': config.get('use_rgb', True),
            **policy_config,
        }
        if is_stage:
            wandb_config['stage'] = stage_idx
            wandb_config['total_stages'] = total_stages
        wandb.init(project=config.get('wandb_project', 'act-training'), name=run_name, config=wandb_config)

    # Create policy
    policy = make_policy(policy_class, policy_config)

    # Load weights from previous stage (sequential training)
    if prev_checkpoint is not None and os.path.exists(prev_checkpoint):
        print(f"{stage_label}Loading weights from previous stage: {prev_checkpoint}")
        policy.load_state_dict(torch.load(prev_checkpoint, map_location="cpu"))

    policy.cuda()
    optimizer = policy.configure_optimizers()

    # Resume support (only for non-sequential training)
    resume_state_path = os.path.join(ckpt_dir, f"resume_state{stage_suffix}.ckpt")
    start_epoch = 0

    if resume and not is_stage:
        ckpt_to_load = None

        # Priority 1: explicit --resume_ckpt
        if resume_ckpt:
            ckpt_to_load = resume_ckpt
            if not os.path.isabs(ckpt_to_load):
                ckpt_to_load = os.path.join(ckpt_dir, ckpt_to_load)

        # Priority 2: resume_state.ckpt (has optimizer state + epoch)
        elif os.path.exists(resume_state_path):
            ckpt_to_load = resume_state_path

        # Priority 3: policy_last.ckpt or policy_best.ckpt
        else:
            for candidate in ("policy_last.ckpt", "policy_best.ckpt"):
                p = os.path.join(ckpt_dir, candidate)
                if os.path.exists(p):
                    ckpt_to_load = p
                    break

            # Priority 4: latest epoch checkpoint
            if ckpt_to_load is None:
                pattern = os.path.join(ckpt_dir, f"policy_epoch_*_seed_{seed}.ckpt")
                files = glob.glob(pattern)
                best_epoch_found = -1
                for fp in files:
                    m = re.search(r"policy_epoch_(\d+)_seed_", os.path.basename(fp))
                    if m and int(m.group(1)) > best_epoch_found:
                        best_epoch_found = int(m.group(1))
                        ckpt_to_load = fp
                if best_epoch_found >= 0:
                    start_epoch = best_epoch_found + 1

        if ckpt_to_load is None or not os.path.exists(ckpt_to_load):
            raise FileNotFoundError(f"[resume] no checkpoint found under ckpt_dir={ckpt_dir}")

        print(f"[resume] loading from: {ckpt_to_load}")
        obj = torch.load(ckpt_to_load, map_location="cuda")

        if isinstance(obj, dict) and ("policy" in obj or "optimizer" in obj):
            # Full resume state: policy + optimizer + epoch
            if "policy" in obj:
                policy.load_state_dict(obj["policy"])
            if "optimizer" in obj:
                try:
                    optimizer.load_state_dict(obj["optimizer"])
                except Exception as e:
                    print(f"[resume] optimizer state load failed (continuing with fresh optimizer): {e}")
            if "epoch" in obj and isinstance(obj["epoch"], int):
                start_epoch = max(start_epoch, obj["epoch"] + 1)
        else:
            # Raw state_dict (from policy_best.ckpt etc.)
            policy.load_state_dict(obj)

        print(f"[resume] starting from epoch {start_epoch}")
        if start_epoch >= num_epochs:
            print(f"[resume] start_epoch={start_epoch} >= num_epochs={num_epochs}, nothing to do.")
            return (start_epoch - 1, np.inf, deepcopy(policy.state_dict()))

    # Initialize MuJoCo visualizers (optional, for training visualization).
    # Token-level (sonic) body actions are latents, not joints — skip viz entirely.
    body_viz, hand_viz, hand_viz_right = None, None, None
    if channel == "sonic":
        print(f'{stage_label}Token-level (sonic): skipping MuJoCo visualizers')
    else:
        try:
            from sim_viz.visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths
            paths = get_default_paths()
            body_viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])
            if hand_side == "both":
                hand_viz = HandVisualizer(paths['left_hand_xml'], hand_side='left')
                hand_viz_right = HandVisualizer(paths['right_hand_xml'], hand_side='right')
            else:
                hand_viz = HandVisualizer(paths[f'{hand_side}_hand_xml'], hand_side=hand_side)
            print(f'{stage_label}Visualizers initialized (body + {hand_side} hand)')
        except Exception as e:
            print(f'{stage_label}Visualizers not available: {e}')

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    save_every_k_epochs = 1000
    visualize_every_k_epochs = 500

    val_data_viz = next(iter(val_dataloader))
    train_data_viz = next(iter(train_dataloader))

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\n{stage_label}Epoch {epoch}')

        # Validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ' '.join(f'{k}: {v.item():.3f}' for k, v in epoch_summary.items())
        print(summary_string)

        # Training
        policy.train()
        optimizer.zero_grad()
        epoch_train_dicts = []
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_dicts.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(epoch_train_dicts)
        train_history.append(epoch_summary)
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ' '.join(f'{k}: {v.item():.3f}' for k, v in epoch_summary.items())
        print(summary_string)

        # Wandb logging
        if use_wandb:
            log_dict = {'epoch': epoch}
            if is_stage:
                log_dict['stage'] = stage_idx
            for k, v in epoch_summary.items():
                log_dict[f'train/{k}'] = v.item()
            log_dict['val/loss'] = epoch_val_loss.item() if hasattr(epoch_val_loss, 'item') else epoch_val_loss
            for k, v in validation_history[-1].items():
                log_dict[f'val/{k}'] = v.item()
            wandb.log(log_dict)

        # Periodic epoch checkpoint
        if epoch % save_every_k_epochs == 0:
            ckpt_name = f'policy{stage_suffix}_epoch_{epoch}_seed_{seed}.ckpt'
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, ckpt_name))
            plot_history(train_history, validation_history, epoch, ckpt_dir,
                         f"{seed}{stage_suffix}")

        # Resume state checkpoint
        if epoch % max(1, resume_save_every) == 0 or epoch == num_epochs - 1:
            try:
                torch.save(
                    {"epoch": int(epoch), "policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
                    resume_state_path,
                )
            except Exception as e:
                print(f"[resume] failed to write {os.path.basename(resume_state_path)}: {e}")

        # Periodic visualization
        if epoch % visualize_every_k_epochs == 0:
            if body_viz is not None and norm_stats is not None:
                with torch.inference_mode():
                    policy.eval()
                    val_split = f'val_s{stage_idx}' if is_stage else 'val'
                    train_split = f'train_s{stage_idx}' if is_stage else 'train'

                    a_hat_val = forward_pass(val_data_viz, policy, pred_action=True)
                    val_video = visualize_predictions(
                        val_data_viz, a_hat_val, norm_stats, body_viz, hand_viz,
                        ckpt_dir, epoch, split=val_split,
                        action_body_dim=action_body_dim, hand_side=hand_side,
                        hand_viz_right=hand_viz_right)
                    del a_hat_val

                    a_hat_train = forward_pass(train_data_viz, policy, pred_action=True)
                    train_video = visualize_predictions(
                        train_data_viz, a_hat_train, norm_stats, body_viz, hand_viz,
                        ckpt_dir, epoch, split=train_split,
                        action_body_dim=action_body_dim, hand_side=hand_side,
                        hand_viz_right=hand_viz_right)
                    del a_hat_train

                if use_wandb:
                    log_payload = {}
                    if val_video is not None:
                        log_payload['val/visualization'] = wandb.Video(val_video)
                    if train_video is not None:
                        log_payload['train/visualization'] = wandb.Video(train_video)
                    if log_payload:
                        wandb.log(log_payload)
                gc.collect()

    # Save final checkpoints
    last_name = f'policy{stage_suffix}_last.ckpt'
    torch.save(policy.state_dict(), os.path.join(ckpt_dir, last_name))

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    best_name = f'policy{stage_suffix}_epoch_{best_epoch}_seed_{seed}.ckpt'
    torch.save(best_state_dict, os.path.join(ckpt_dir, best_name))
    print(f'{stage_label}Training finished: seed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    plot_history(train_history, validation_history, num_epochs, ckpt_dir, f"{seed}{stage_suffix}")

    if use_wandb:
        wandb.log({
            'best_epoch': best_epoch,
            'best_val_loss': min_val_loss.item() if hasattr(min_val_loss, 'item') else min_val_loss,
        })
        wandb.finish()

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        fig = plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        epochs = np.arange(len(train_history))
        plt.plot(epochs, train_values, label='train')
        plt.plot(epochs, val_values, label='validation')
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close(fig)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_root', type=str, required=True, help='Root directory for checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Checkpoint directory (for resume)')
    parser.add_argument('--ckpt_prefix', type=str, default='',
                        help='Prefix for checkpoint folder name. Example: --ckpt_prefix exp1 -> exp1_YYYYmmdd_HHMMSS')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from --ckpt_dir')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Specific checkpoint to resume from (path or filename under ckpt_dir)')
    parser.add_argument('--resume_save_every', type=int, default=20,
                        help='Save resume state every N epochs (default: 20)')

    parser.add_argument('--policy_class', type=str, required=True, help='ACT or CNNMLP')
    parser.add_argument('--task_name', type=str, required=True, help='Task name (used for checkpoint organization)')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)

    parser.add_argument('--dataset_path', type=str, nargs='+', required=True,
                        help='Path(s) to HDF5 dataset file(s)')
    parser.add_argument('--hand_side', type=str, default='left', choices=['left', 'right', 'both'],
                        help='Which hand(s) to control')
    parser.add_argument('--channel', type=str, default='twist2', choices=['twist2', 'sonic'],
                        help='Data channel: twist2 (joint action_body) or sonic (token action_token)')


    # ACT-specific
    parser.add_argument('--kl_weight', type=int, help='KL weight for CVAE')
    parser.add_argument('--chunk_size', type=int, help='Action chunk size (num_queries)')
    parser.add_argument('--hidden_dim', type=int, help='Transformer hidden dimension')
    parser.add_argument('--dim_feedforward', type=int, help='Transformer feedforward dimension')
    parser.add_argument('--temporal_agg', action='store_true')

    parser.add_argument('--no_rgb', action='store_true',
                        help='State-only training (no visual observations)')

    # Wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='act-training')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    # Sequential training
    parser.add_argument('--sequential_training', action='store_true',
                        help='Train on datasets sequentially instead of mixing')
    parser.add_argument('--epochs_per_dataset', type=int, nargs='+',
                        help='Epochs per dataset for sequential training')
    parser.add_argument('--sequential_unified_stats', action='store_true',
                        help='Use unified normalization stats across all sequential stages')

    parser.add_argument('--val_robot_only', action='store_true',
                        help="Use only robot dataset episodes for validation")

    main(vars(parser.parse_args()))
