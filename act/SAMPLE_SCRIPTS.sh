### to process data 

# robot data
python convert_to_hdf5.py --dataset_dirs ROBOT_DATA_DIR1 ROBOT_DATA_DIR2 --output ROBOT_DATASET_PATH.hdf5 --state_body_31d

# human data
python scripts/batch_convert_hand_tracking.py --dataset_dir HUMAN_DATA_DIR # can skip
python scripts/shift_state_action.py --dataset_dir HUMAN_DATA_DIR
python convert_to_hdf5.py --dataset_dirs HUMAN_DATA_DIR --output HUMAN_DATASET_PATH.hdf5 --state_body_31d

# after conversion, use eval_policy.sh option (2) to replay the data in sim. Add --sim_only --sim_save_vid to save video. 

### to train policy

# sequential 
# dataset order matters! argument order = training order
MUJOCO_GL=egl python imitate_episodes.py \
  --task_name pickball --dataset_path HUMAN_DATASET_PATH.hdf5 ROBOT_DATASET_PATH.hdf5 --ckpt_root ckpt/ \
  --sequential_training \
  --epochs_per_dataset 1000 1000 \
  --policy_class ACT --batch_size 16 --seed 0 \
  --num_epochs 2000 --lr 2e-5 --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --state_body_dim 31 --hand_side right\
  --wandb --wandb_project act-training --wandb_run_name squatandpick0115_rgb_0115_robothuman31D_seq_chunk100

# mix
MUJOCO_GL=egl python imitate_episodes.py \
  --task_name pickball --dataset_path HUMAN_DATASET_PATH.hdf5 ROBOT_DATASET_PATH.hdf5 --ckpt_root ckpt/ \
  --policy_class ACT --batch_size 16 --seed 0 \
  --num_epochs 2000 --lr 2e-5 --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --state_body_dim 31 --hand_side right\
  --wandb --wandb_project act-training --wandb_run_name picktowel0112_rgb_0113_robothuman31D_chunk100 

# robot data only, then just pass in the robot dataset path