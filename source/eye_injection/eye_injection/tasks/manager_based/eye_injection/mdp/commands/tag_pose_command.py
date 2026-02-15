# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import (
    combine_frame_transforms,
    convert_camera_frame_orientation_convention,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera
    from torch import Tensor

    from .commands_cfg import TagPoseCommandCfg


@torch.jit.script
def combine_tag_poses_ids(poses: Tensor, ids: Tensor) -> Tensor:
    """Combine the tag pose commands with their corresponding IDs.

    Each pose-id pair is combined in the form (id, pose).

    Args:
        poses: Tag poses. Shape is (N, nT * 7).
        ids: Tag ids. Shape is (N, nT).

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


class TagPoseCommand(CommandTerm):
    """Tag pose command term for retargeting robot pose commands.

    The command generator generates poses by retargeting pose commands that were generated for the
    robot's EE relative to its base. Pose commands are retargeted such that they represent the pose
    of AprilTags relative to the desired camera frame.

    This pose transformation requires the following transforms:
        1) Base <- EE (target)
        2) EE <- Camera
        3) Base <- Tag_i
    using these 3 transforms, the new target pose is generated for Tag_i as the following:
        - Camera (target) <- Tag_i = inv(Base <- EE (target) <- Camera) * (Base <- Tag_i)

    The command generator is configured to work alongside the PoseCommand generator.
    """

    cfg: TagPoseCommandCfg
    """Configuration for the tag command generator."""

    def __init__(self, cfg: TagPoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the tag command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check number of tags and ids
        assert len(self.cfg.tag_prim_names) > 0, (
            "No tag primitive names passed. Received an empty list."
        )
        assert len(self.cfg.tag_prim_names) == len(self.cfg.tag_ids), (
            "Each tag primitve must have a matching ID. Got mismatched lists."
        )
        self.n_tags = len(self.cfg.tag_prim_names)

        # extract the relative pose of the camera asset relative to the end-effector
        # (assumed to be fixed)
        self.pose_cam_ee = self._get_relative_camera_pose(env)
        self.pose_cam_ee = torch.tile(self.pose_cam_ee, (self.n_tags, 1))  # (nT * N, 7)

        # extract the tag assets' relative poses to the base (assumed to be fixed)
        self.pose_tag_base = self._get_relative_tag_poses()
        self.pose_tag_base = torch.cat(self.pose_tag_base, dim=0)  # (nT * N, 7)

        # create buffers to store the command
        # -- commands: ([tag_id_0, tag_pose_0], ..., [tag_id_nT, tag_pose_nT])
        self.id_command = torch.tile(
            torch.tensor(self.cfg.tag_ids, dtype=torch.float32, device=self.device),
            (env.num_envs, 1),
        )  # (N, nT)
        self.pose_command = torch.zeros(
            self.num_envs, self.n_tags * 7, device=self.device
        )
        for i in range(self.n_tags):
            self.pose_command[:, i * 7 + 3] = 1.0  # valid unit quaternions

        # -- no metrics to track currently

    def __str__(self) -> str:
        msg = "TagPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> Tensor:
        """The tag ids and poses in the desired camera frames. Shape is (num_envs, nT * 8)."""
        return combine_tag_poses_ids(self.pose_command, self.id_command)

    @property
    def target_pose_command(self) -> Tensor:
        """The latest generated pose command. Shape is (num_envs, 7)."""
        return self._env.command_manager.get_command(self.cfg.pose_command_name)[:, 1:]

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
        # get latest generated pose command (Base <- EE (target))
        pose_ee_base = self.target_pose_command
        pose_ee_base = torch.tile(pose_ee_base, (self.n_tags, 1))  # (nT * N, 7)

        # compute target pose for camera (Base <- Camera (target))
        pos_cam_base, rot_cam_base = combine_frame_transforms(
            pose_ee_base[:, :3],
            pose_ee_base[:, 3:],
            self.pose_cam_ee[:, :3],
            self.pose_cam_ee[:, 3:],
        )

        # compute tag pose relative to camera (Camera (target) <- Tag)
        pos_tag_cam, rot_tag_cam = subtract_frame_transforms(
            pos_cam_base,
            rot_cam_base,
            self.pose_tag_base[:, :3],
            self.pose_tag_base[:, 3:],
        )

        # update stored pose command
        pose_tag_cam = torch.cat([pos_tag_cam, rot_tag_cam], dim=-1)
        self.pose_command = (
            pose_tag_cam.reshape(self.n_tags, self.num_envs, 7)
            .transpose(0, 1)
            .reshape(self.num_envs, self.n_tags * 7)
        )

    """
    Private helper functions.
    """

    def _get_relative_camera_pose(self, env: ManagerBasedRLEnv) -> Tensor:
        """Get the relative pose of the camera asset with respect to the end-effector asset.

        Args:
            env: The environment object.

        Returns:
            The relative pose of the camera. Shape is (N, 7).
        """
        # Retrieve the camera sensor and offset configuration from environment
        asset: Camera = env.scene[self.cfg.camera_asset_name]
        offset = asset.cfg.offset

        # Get camera position and orientation in ROS convention (forward axis +Z)
        pos = torch.tensor(offset.pos)
        rot = torch.tensor(offset.rot)
        if offset.convention != "ros":
            rot = convert_camera_frame_orientation_convention(
                rot, origin=offset.convention, target="ros"
            )

        # Repeat camera pose num_envs times
        pose = torch.tile(torch.cat([pos, rot], dim=0), (env.num_envs, 1))
        pose = pose.to(device=self.device)
        return pose

    def _get_relative_tag_poses(self) -> list[Tensor]:
        """Get the relative poses of the tag assets with respect to the body asset.

        Returns:
            A list containing the poses of the tag assets with respect to the base asset. Length
                equal to the number of tag assets. Shape of each entry is (N, 7).
        """
        # extract the reference prims
        ref_prims = sim_utils.find_matching_prims(self.cfg.pose_source_prim_name)

        # extract relative poses for each tag prim
        poses = []
        for tag_prim_name in self.cfg.tag_prim_names:
            # extract the tag prims
            tag_prims = sim_utils.find_matching_prims(tag_prim_name)
            assert len(tag_prims) == len(ref_prims), (
                f"Target and reference prims must have matching lengths, got {len(tag_prims)}"
                f" and {len(ref_prims)}."
            )

            # extract the individual relative poses across environment instances
            pos, quat = [], []
            for t_prim, r_prim in zip(tag_prims, ref_prims):
                pos_i, quat_i = sim_utils.resolve_prim_pose(t_prim, ref_prim=r_prim)
                pos.append(torch.tensor(pos_i))
                quat.append(torch.tensor(quat_i))
            pos = torch.stack(pos, dim=0).to(device=self.device)
            quat = torch.stack(quat, dim=0).to(device=self.device)
            poses.append(torch.cat([pos, quat], dim=-1))
        return poses
