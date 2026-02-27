# # 0124 sequential training on pick bread task with only bread object human data, robots train for longer, unified normalization stats
# CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python imitate_episodes.py \
#   --task_name pickball --dataset_path data/20250120_pick_bread_human_31D.hdf5 data/20250117_pick_bread_robot_31D.hdf5 --ckpt_root ckpt/pickbread \
#   --sequential_training --val_robot_only --sequential_unified_stats \
#   --epochs_per_dataset 1000 4000 \
#   --policy_class ACT --batch_size 16 --seed 0 --ckpt_prefix sequential_breadhuman_chunk200_robotlonger4k_unifiedstats\
#   --num_epochs 5000 --lr 2e-5 --kl_weight 10 --chunk_size 200 --hidden_dim 512 --dim_feedforward 3200 --state_body_dim 31 --hand_side right\
#   --wandb --wandb_project act-training --wandb_run_name pickbread0120_rgb_0124_sequential_breadhuman_chunk200_robotlonger4k_unifiedstats

# # 0124 sequential training on pick bread task with earlier fast human data, robots train for longer, unified normalization stats
# CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python imitate_episodes.py \
#   --task_name pickball --dataset_path data/20250117_pick_bread_human_fast_31D.hdf5 data/20250117_pick_bread_robot_31D.hdf5 --ckpt_root ckpt/pickbread \
#   --sequential_training --val_robot_only --sequential_unified_stats \
#   --epochs_per_dataset 1000 4000 \
#   --policy_class ACT --batch_size 16 --seed 0 --ckpt_prefix sequential_fasthuman_chunk200_robotlonger4k_unifiedstats\
#   --num_epochs 5000 --lr 2e-5 --kl_weight 10 --chunk_size 200 --hidden_dim 512 --dim_feedforward 3200 --state_body_dim 31 --hand_side right\
#   --wandb --wandb_project act-training --wandb_run_name pickbread0120_rgb_0124_sequential_fasthuman_chunk200_robotlonger4k_unifiedstats

# # 0123 sequential training on pick bread task with only bread object human data, robots train for longer
# CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python imitate_episodes.py \
#   --task_name pickball --dataset_path data/20250120_pick_bread_human_31D.hdf5 data/20250117_pick_bread_robot_31D.hdf5 --ckpt_root ckpt/pickbread \
#   --sequential_training --val_robot_only \
#   --epochs_per_dataset 1000 4000 \
#   --policy_class ACT --batch_size 16 --seed 0 --ckpt_prefix sequential_breadhuman_chunk200_robotlonger4k\
#   --num_epochs 5000 --lr 2e-5 --kl_weight 10 --chunk_size 200 --hidden_dim 512 --dim_feedforward 3200 --state_body_dim 31 --hand_side right\
#   --wandb --wandb_project act-training --wandb_run_name pickbread0120_rgb_0124_sequential_breadhuman_chunk200_robotlonger4k

# MUJOCO_GL=egl python imitate_episodes.py \
#   --task_name pickball --dataset_path data/20250125_opendoor_robot_31D.hdf5 --ckpt_root ckpt/open_door \
#   --policy_class ACT --batch_size 16 --seed 0 --ckpt_prefix chunk400 \
#   --num_epochs 8000 --lr 2e-5 --kl_weight 10 --chunk_size 400 --hidden_dim 512 --dim_feedforward 3200 --state_body_dim 31 --hand_side both \
#   --wandb --wandb_project act-training --wandb_run_name opendoor0128_chunk400

# MUJOCO_GL=egl python imitate_episodes.py \
#   --task_name pickball --dataset_path data/20250125_opendoor_robot_31D.hdf5 --ckpt_root ckpt/open_door \
#   --policy_class ACT --batch_size 16 --seed 0 --ckpt_prefix chunk500 \
#   --num_epochs 8000 --lr 2e-5 --kl_weight 10 --chunk_size 500 --hidden_dim 512 --dim_feedforward 3200 --state_body_dim 31 --hand_side both \
#   --wandb --wandb_project act-training --wandb_run_name opendoor0128_chunk500

MUJOCO_GL=egl python imitate_episodes.py \
  --task_name pickball --dataset_path data/20250129_pickbread_robot_31D.hdf5 --ckpt_root ckpt/pick_bread_new \
  --policy_class ACT --batch_size 16 --seed 0 --ckpt_prefix chunk300 \
  --num_epochs 5000 --lr 2e-5 --kl_weight 10 --chunk_size 300 --hidden_dim 512 --dim_feedforward 3200 --state_body_dim 31 --hand_side both \
  --wandb --wandb_project act-training --wandb_run_name pickbreadnew_chunk300