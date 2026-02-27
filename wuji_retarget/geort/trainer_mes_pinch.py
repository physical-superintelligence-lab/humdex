# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien.core as sapien
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from geort.utils.path import get_human_data
from geort.utils.config_utils import get_config, save_json
from geort.model import FKModel, IKModel
from geort.env.hand import HandKinematicModel
from geort.formatter import HandFormatter
from geort.dataset import RobotKinematicsDataset
from datetime import datetime
from tqdm import tqdm
import os
from pathlib import Path
import math


def merge_dict_list(dl):
    keys = dl[0].keys()
    result = {k: [] for k in keys}
    for data in dl:
        for k in keys:
            result[k].append(data[k])
    result = {k: np.array(v) for k, v in result.items()}
    return result


def format_loss(value):
    return f"{value:.4e}" if math.fabs(value) < 1e-3 else f"{value:.4f}"


def get_float_list_from_np(np_vector):
    float_list = np_vector.tolist()
    float_list = [float(x) for x in float_list]
    return float_list


def generate_current_timestring():
    """Utility Function. Generate a current timestring in the format 'YYYY-MM-DD_HH-MM-SS'."""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class SupervisedRetargetDataset(Dataset):
    """
    Supervised dataset: (human_fingertips_rel_wrist -> robot_qpos_target)
    - Reads fingertips_rel_wrist: [T,5,>=3]
    - Reads qpos target key (default: action_wuji_qpos_target): [T,DOF] or [T,1,DOF]
    - Applies per-finger scaling to fingertip points:
        scaled = human / (human/robot) = human * (robot/human)
    """

    def __init__(
        self,
        npz_path: str,
        qpos_key: str = "action_wuji_qpos_target",
        n: int = 20000,
        per_finger_scale_h_over_r=None,
    ):
        super().__init__()
        pack = np.load(npz_path, allow_pickle=True)

        if "fingertip_names" in pack.files:
            print("[Dataset] fingertip_names:", pack["fingertip_names"])

        # breakpoint()


        if "fingertips_rel_wrist" not in pack.files:
            raise KeyError(
                f"[SupervisedRetargetDataset] 'fingertips_rel_wrist' not found in {npz_path}. "
                f"Available keys: {pack.files}"
            )
        if qpos_key not in pack.files:
            raise KeyError(
                f"[SupervisedRetargetDataset] qpos_key='{qpos_key}' not found in {npz_path}. "
                f"Available keys: {pack.files}"
            )

        human_points_raw = pack["fingertips_rel_wrist"]  # [T, 5, 3] (or [T,5,>=3])
        if human_points_raw.ndim != 3 or human_points_raw.shape[1] != 5 or human_points_raw.shape[2] < 3:
            raise ValueError(
                f"[SupervisedRetargetDataset] Expect fingertips_rel_wrist shape [T,5,>=3], got {human_points_raw.shape}"
            )

        qpos = pack[qpos_key]
        # allow [T,1,DOF]
        if qpos.ndim == 3 and qpos.shape[1] == 1:
            qpos = qpos[:, 0, :]
        if qpos.ndim != 2:
            raise ValueError(
                f"[SupervisedRetargetDataset] Expect {qpos_key} shape [T,DOF] (or [T,1,DOF]), got {qpos.shape}"
            )

        T = human_points_raw.shape[0]
        if qpos.shape[0] != T:
            raise ValueError(
                f"[SupervisedRetargetDataset] Time length mismatch: fingertips T={T} vs {qpos_key} T={qpos.shape[0]}"
            )

        self.human_xyz = human_points_raw[:, :, :3].astype(np.float32)  # [T,5,3]
        self.qpos = qpos.astype(np.float32)  # [T,DOF]
        self.T = T
        self.n = int(n)
        self.qpos_key = qpos_key

        if per_finger_scale_h_over_r is None:
            # default: no scaling
            self.scale = None
        else:
            s = np.asarray(per_finger_scale_h_over_r, dtype=np.float32)
            if s.shape != (5,):
                raise ValueError(f"[SupervisedRetargetDataset] per_finger_scale_h_over_r must have shape (5,), got {s.shape}")
            self.scale = s  # human/robot

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        t = np.random.randint(0, self.T)
        point = self.human_xyz[t]  # [5,3]
        if self.scale is not None:
            # scaled = human / (human/robot)
            point = point / self.scale[:, None]
        qpos = self.qpos[t]  # [DOF]
        return torch.from_numpy(point), torch.from_numpy(qpos)


class GeoRTTrainer:
    def __init__(self, config):
        self.config = config
        self.hand = HandKinematicModel.build_from_config(self.config)

    def get_robot_kinematics_dataset(self):
        dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        if not os.path.exists(dataset_path):
            _ = self.generate_robot_kinematics_dataset(n_total=100000, save=True)
            dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)

        keypoint_names = self.get_keypoint_info()["link"]
        return RobotKinematicsDataset(dataset_path, keypoint_names=keypoint_names)

    def get_robot_kinematics_dataset_path(self, postfix=False):
        data_name = self.config["name"]
        out = f"data/{data_name}"
        if postfix:
            out += ".npz"
        return out

    def get_keypoint_info(self):
        keypoint_links = []
        keypoint_offsets = []
        keypoint_joints = []
        keypoint_human_ids = []

        joint_order = self.config["joint_order"]

        for info in self.config["fingertip_link"]:
            keypoint_links.append(info["link"])
            keypoint_offsets.append(info["center_offset"])
            keypoint_human_ids.append(info["human_hand_id"])

            kp_joint = []
            for joint in info["joint"]:
                kp_joint.append(joint_order.index(joint))
            keypoint_joints.append(kp_joint)

        return {
            "link": keypoint_links,
            "offset": keypoint_offsets,
            "joint": keypoint_joints,
            "human_id": keypoint_human_ids,
        }

    def generate_robot_kinematics_dataset(self, n_total=100000, save=True):
        info = self.get_keypoint_info()
        self.hand.initialize_keypoint(keypoint_link_names=info["link"], keypoint_offsets=info["offset"])

        joint_range_low, joint_range_high = self.hand.get_joint_limit()
        joint_range_low = np.array(joint_range_low)
        joint_range_high = np.array(joint_range_high)

        all_data_qpos = []
        all_data_keypoint = []

        for _ in tqdm(range(n_total)):
            qpos = (
                np.random.uniform(0, 1, len(joint_range_low))
                * (joint_range_high - joint_range_low)
                + joint_range_low
            )
            keypoint = self.hand.keypoint_from_qpos(qpos)
            all_data_qpos.append(qpos)
            all_data_keypoint.append(keypoint)

        all_data_keypoint = merge_dict_list(all_data_keypoint)
        dataset = {"qpos": all_data_qpos, "keypoint": all_data_keypoint}

        if save:
            os.makedirs("data", exist_ok=True)
            np.savez(self.get_robot_kinematics_dataset_path(), **dataset)

        return dataset

    def get_fk_checkpoint_path(self):
        name = self.config["name"]
        os.makedirs("checkpoint", exist_ok=True)
        return f"checkpoint/fk_model_{name}.pth"

    def get_robot_neural_fk_model(self, force_train=False):
        """
        FK: normalized qpos -> keypoints
        """
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()
        qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)

        fk_model = FKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()
        fk_checkpoint_path = self.get_fk_checkpoint_path()

        if os.path.exists(fk_checkpoint_path) and not force_train:
            fk_model.load_state_dict(torch.load(fk_checkpoint_path, weights_only=True))
        else:
            print("Train Neural Forward Kinematics (FK) from Scratch")
            fk_dataset = self.get_robot_kinematics_dataset()
            fk_dataloader = DataLoader(fk_dataset, batch_size=256, shuffle=True)
            fk_optim = optim.Adam(fk_model.parameters(), lr=5e-4)
            criterion_fk = nn.MSELoss()

            for epoch in range(500):
                all_fk_error = 0.0
                for batch_idx, batch in enumerate(fk_dataloader):
                    keypoint = batch["keypoint"].cuda().float()
                    qpos = batch["qpos"].cuda().float()
                    qpos = qpos_normalizer.normalize_torch(qpos)

                    pred_keypoint = fk_model(qpos)

                    fk_optim.zero_grad()
                    loss = criterion_fk(pred_keypoint, keypoint)
                    loss.backward()
                    fk_optim.step()

                    all_fk_error += loss.item()

                avg_fk_error = all_fk_error / (batch_idx + 1)
                print(f"Neural FK Training Epoch: {epoch}; Training Loss: {avg_fk_error}")

            torch.save(fk_model.state_dict(), fk_checkpoint_path)

        fk_model.eval()
        return fk_model

    @staticmethod
    def pinch_loss_from_points(point: torch.Tensor,
                            embedded_point: torch.Tensor,
                            thresh: float = 0.015,
                            eps: float = 1e-7,
                            thumb_idx: int = 0,
                            pinky_idx: int = 4, ) -> torch.Tensor:
        """
        Only compute pinch constraints between thumb and other fingertips.

        point:          [B,5,3] scaled human fingertips
        embedded_point: [B,5,3] robot fingertips (FK(IK(point)))
        thresh:         pinch trigger threshold in input space (meters)
        """
        loss = torch.zeros((), device=point.device, dtype=point.dtype)

        # thumb vs others
        for j in range(point.size(1)):
            if j == thumb_idx:
                continue
            if j == pinky_idx:
                continue

            dist_in = point[:, thumb_idx, :] - point[:, j, :]             # [B,3]
            mask = (torch.norm(dist_in, dim=-1) < thresh).float()         # [B]

            dist_emb = embedded_point[:, thumb_idx, :] - embedded_point[:, j, :]  # [B,3]
            e_dist2 = (dist_emb ** 2).sum(dim=-1)                         # [B]

            loss_ij = (mask * e_dist2).sum() / (mask.sum() + eps)
            loss = loss + loss_ij

        return loss


    def train(self, human_data_path, **kwargs):
        """
        Total loss:
          loss = w_mse * MSE(ik_model(point), normalize(qpos_gt)) + w_pinch * pinch_loss(point, fk_model(ik_pred))
        """
        # --------------------
        # Models
        # --------------------
        fk_model = self.get_robot_neural_fk_model(force_train=kwargs.get("force_train_fk", False))
        ik_model = IKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()
        ik_model.train()

        ik_optim = optim.AdamW(ik_model.parameters(), lr=kwargs.get("lr", 1e-4))

        # --------------------
        # Workspace / ckpt
        # --------------------
        exp_tag = kwargs.get("tag", "")
        n_epoch = kwargs.get("epoch", 500)
        save_every = kwargs.get("save_every", 10)

        hand_model_name = self.config["name"]
        ckpt_root = Path(kwargs.get("ckpt_root", "/data/jiajunxu/models/hand_retarget"))

        run_name = f"{hand_model_name}_{generate_current_timestring()}"
        if exp_tag != "":
            run_name += f"_{exp_tag}"

        save_dir = ckpt_root / run_name
        last_save_dir = ckpt_root / f"{hand_model_name}_last"
        save_dir.mkdir(parents=True, exist_ok=True)
        last_save_dir.mkdir(parents=True, exist_ok=True)

        # Save config (including joint info + scaling + loss weights)
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()
        export_config = self.config.copy()
        export_config["joint"] = {
            "lower": get_float_list_from_np(joint_lower_limit),
            "upper": get_float_list_from_np(joint_upper_limit),
        }

        # --------------------
        # Per-finger scaling (human/robot)
        # --------------------
        WUJI_FINGER_TIP_SCALING = [
            14.446 / 12.92,
            20.135 / 18.17,
            19.511 / 18.33,
            19.053 / 17.34,
            18.633 / 14.89,
        ]

        export_config["human_point_scaling"] = {
            "type": "per_finger_divide_human_over_robot",
            "values": [float(x) for x in WUJI_FINGER_TIP_SCALING],
            "note": "scaled_human = human / (human/robot)",
        }

        # loss weights & pinch thresh
        w_mse = float(kwargs.get("w_mse", 1.0))
        w_pinch = float(kwargs.get("w_pinch", 1.0))
        pinch_thresh = float(kwargs.get("pinch_thresh", 0.015))
        export_config["loss"] = {"w_mse": w_mse, "w_pinch": w_pinch, "pinch_thresh": pinch_thresh}

        save_json(export_config, save_dir / "config.json")
        save_json(export_config, last_save_dir / "config.json")

        # --------------------
        # Normalizer (match FK convention)
        # --------------------
        qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)

        # --------------------
        # Dataset
        # --------------------
        qpos_key = kwargs.get("qpos_key", "action_wuji_qpos_target")
        n_samples = int(kwargs.get("n_samples", 20000))
        batch_size = int(kwargs.get("batch_size", 2048))

        ds = SupervisedRetargetDataset(
            human_data_path,
            qpos_key=qpos_key,
            n=n_samples,
            per_finger_scale_h_over_r=WUJI_FINGER_TIP_SCALING,
        )
        print(f"[Dataset] Using qpos key: {ds.qpos_key}")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        # --------------------
        # Losses
        # --------------------
        mse_criterion = nn.MSELoss()

        for epoch in range(n_epoch):
            running_total = 0.0
            running_mse = 0.0
            running_pinch = 0.0

            for batch_idx, (point, qpos_gt) in enumerate(dl):
                point = point.cuda().float()      # [B,5,3] scaled already
                qpos_gt = qpos_gt.cuda().float()  # [B,DOF]

                # normalize GT to match IK output space (assumed normalized)
                qpos_gt_norm = qpos_normalizer.normalize_torch(qpos_gt)

                # IK forward
                joint_pred = ik_model(point)  # [B,DOF] (assumed normalized)

                # MSE loss (supervised)
                mse_loss = mse_criterion(joint_pred, qpos_gt_norm)

                # Pinch loss uses FK embedding of predicted joints
                embedded_point = fk_model(joint_pred)  # [B,5,3]
                pinch_loss = self.pinch_loss_from_points(point, embedded_point, thresh=pinch_thresh)

                loss = w_mse * mse_loss + w_pinch * pinch_loss

                ik_optim.zero_grad()
                loss.backward()
                ik_optim.step()

                running_total += float(loss.item())
                running_mse += float(mse_loss.item())
                running_pinch += float(pinch_loss.item())

                if batch_idx % 50 == 0:
                    denom = max(1, batch_idx + 1)
                    print(
                        f"Epoch {epoch} | "
                        f"Total: {format_loss(running_total/denom)} | "
                        f"MSE: {format_loss(running_mse/denom)} | "
                        f"Pinch: {format_loss(running_pinch/denom)}"
                    )

            # checkpoints
            if (epoch % save_every == 0) or (epoch == n_epoch - 1):
                torch.save(ik_model.state_dict(), save_dir / f"epoch_{epoch}.pth")
                torch.save(ik_model.state_dict(), last_save_dir / f"epoch_{epoch}.pth")

            torch.save(ik_model.state_dict(), save_dir / "last.pth")
            torch.save(ik_model.state_dict(), last_save_dir / "last.pth")

        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-hand", type=str, default="allegro")
    parser.add_argument("-human_data", type=str, default="human")
    parser.add_argument("-ckpt_tag", type=str, default="")

    # supervised dataset options
    parser.add_argument("--qpos_key", type=str, default="action_wuji_qpos_target")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=2048)

    # training options
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--ckpt_root", type=str, default="./checkpoint")

    # loss weights
    parser.add_argument("--w_mse", type=float, default=1.0)
    parser.add_argument("--w_pinch", type=float, default=0.08)
    parser.add_argument("--pinch_thresh", type=float, default=0.001)

    # fk
    parser.add_argument("--force_train_fk", action="store_true")

    args = parser.parse_args()

    config = get_config(args.hand)
    trainer = GeoRTTrainer(config)

    human_data_path = get_human_data(args.human_data)
    print("Training with human data:", human_data_path.as_posix())

    trainer.train(
        human_data_path.as_posix(),
        tag=args.ckpt_tag,
        qpos_key=args.qpos_key,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        epoch=args.epoch,
        save_every=args.save_every,
        ckpt_root=args.ckpt_root,
        w_mse=args.w_mse,
        w_pinch=args.w_pinch,
        pinch_thresh=args.pinch_thresh,
        force_train_fk=args.force_train_fk,
    )
