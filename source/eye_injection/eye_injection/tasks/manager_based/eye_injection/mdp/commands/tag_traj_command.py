# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import CommandTerm

import eye_injection.tasks.utils.isaac.common as utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from torch import Tensor

    from .commands_cfg import TagTrajCommandCfg

# region Command


class TagTrajCommand(CommandTerm):
    """Tag trajectory command term for retargeting robot state commands.

    The command generator retargets state commands that were generated for the robot's EE relative
    to its base. Poses are retargeted such that they represent the pose of the desired camera frame
    relative to the AprilTags. Twist is retargeted from the end-effector to the camera.

    To retarget poses, the following transforms are required:
        1) Base <- EE (target)
        2) EE <- Camera
        3) Base <- Tag_i
    using these 3 transforms, the new target pose is generated for Tag_i as the following:
        - Tag_i <- Camera (target) = inv(Base <- Tag_i) * (Base <- EE (target) <- Camera)

    The command generator is configured to work alongside the `TrajSmCommand` command generator.
    """

    cfg: TagTrajCommandCfg
    """Configuration for the tag command generator."""

    def __init__(self, cfg: TagTrajCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the tag command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check number of tags and ids
        assert len(cfg.tag_prim_names) > 0, "No tag primitive names passed. Received an empty list."
        assert len(cfg.tag_prim_names) == len(cfg.tag_ids), (
            "Each tag primitve must have a matching ID. Got mismatched lists."
        )
        self.n_tags = len(cfg.tag_prim_names)

        # extract the pose of the camera asset relative to the end-effector (assumed to be fixed)
        self.ee_cam_pose = utils.get_camera_relative_pose(env.scene[cfg.camera_asset_name])
        self.ee_cam_pose = self.ee_cam_pose.repeat((env.num_envs, 1)).to(device=self.device)

        # extract the pose of the base relative to the tag assets (assumed to be fixed)
        t_prims, r_prim = cfg.tag_prim_names, cfg.pose_ref_prim_name
        base_tag_pose = [utils.get_prim_relative_pose(t_prim, ref=r_prim) for t_prim in t_prims]
        rot_offset = torch.tensor([-torch.pi, 0.0, 0.0]).repeat((env.num_envs, 1))
        self.tag_base_pose = [
            utils.get_inverse_pose(utils.apply_delta_rot(p, rot_offset)) for p in base_tag_pose
        ]
        self.tag_base_pose = torch.cat(self.tag_base_pose, dim=0)  # (nT * N, 7)

        # create buffers to store the command
        # -- commands: ([cam_twist], [tag_id_0, tag_pose_0], ..., [tag_id_nT, tag_pose_nT])
        self.cam_twist = torch.zeros((env.num_envs, 6), device=self.device)
        self.tag_id = torch.tensor(cfg.tag_ids, device=self.device).repeat((env.num_envs, 1))
        self.tag_cam_pose = torch.zeros((env.num_envs, self.n_tags * 7), device=self.device)

    def __str__(self) -> str:
        msg = "TagTrajCommandCfg:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> Tensor:
        """The current command (cam_twist + tag_i[id + pose] + ... + tag_nT[id + pose]) for each
        environment. Shape is (num_envs, 6 + nT * 8).
        """
        tag_id_pose = _combine_tag_ids_poses(self.tag_id, self.tag_cam_pose)
        return torch.cat([self.cam_twist, tag_id_pose], dim=-1)

    @property
    def target_state_command(self) -> Tensor:
        """The latest generated end-effector state (pose + twist) command. Shape is (num_envs, 13)."""
        return self._env.command_manager.get_command(self.cfg.traj_command_name)[:, 1:]

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics based on the current state."""
        # no metrics to track currently
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command for the specified environments."""
        # resampling is not needed for this command generator
        # commands will be updated when _update_command is called

    def _update_command(self) -> None:
        """Update the command based on the current state."""
        # get latest generated state command. Pose is (Base <- EE (target))
        base_ee_state = self.target_state_command
        base_ee_pose, ee_twist = base_ee_state[:, :7], base_ee_state[:, 7:]

        # compute target pose and twist for camera. Pose is (Base <- Camera (target))
        base_cam_pose = utils.get_combined_pose(base_ee_pose, self.ee_cam_pose)
        base_cam_pose = base_cam_pose.repeat((self.n_tags, 1))  # (nT * N, 7)
        self.cam_twist = utils.transform_twist(self.ee_cam_pose, ee_twist)

        # compute camera pose relative to tags (Tag <- Camera (target))
        tag_cam_pose = utils.get_combined_pose(self.tag_base_pose, base_cam_pose)
        self.tag_cam_pose = (
            tag_cam_pose.reshape(self.n_tags, self.num_envs, 7)
            .transpose(0, 1)
            .reshape(self.num_envs, self.n_tags * 7)
        )


# region JiT Helpers


@torch.jit.script
def _combine_tag_ids_poses(ids: Tensor, poses: Tensor) -> Tensor:
    """Combine the tag IDs with their corresponding pose commands.

    Each id-pose pair is combined in the form (id, pose).

    Args:
        ids: Tag ids. Shape is (N, nT).
        poses: Tag poses. Shape is (N, nT * 7).

    Returns:
        Combined tag poses with ids. Shape is (N, nT * 8).
    """
    n_envs, n_tags = ids.shape
    device = ids.device
    combined = torch.zeros(n_envs, n_tags * 8, dtype=torch.float32, device=device)
    for i in range(n_tags):
        combined[:, i * 8] = ids[:, i]
        combined[:, i * 8 + 1 : (i + 1) * 8] = poses[:, i * 7 : (i + 1) * 7]
    return combined
