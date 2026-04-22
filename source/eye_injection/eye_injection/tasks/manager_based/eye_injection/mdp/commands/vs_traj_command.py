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

    from .commands_cfg import VsTrajCommandCfg


class VsTrajCommand(CommandTerm):
    """Visual servoing trajectory command term for retargeting robot pose commands.

    The command generator retargets pose commands that were generated for the robot's EE relative
    to its base. Poses are retargeted such that they represent the pose of the robot's EE relative
    to the reference primitive directly, removing the dependency on the robot's joint configuration.
    Twist is left unchanged since it's already represented in the EE's local frame. This is done to
    facilitate vision-based control.

    To retarget poses, the following transforms are required:
        1) Base <- EE (target)
        2) Ref_i <- Base
    using these transforms, the new target pose is generated for Ref_i as the following:
        - Ref_i <- EE (target) = (Ref_i <- Base) * (Base <- EE (target))

    The command generator is configured to work alongside the `TrajSmCommand` and `BinaryCommand`
    command generators.
    """

    cfg: VsTrajCommandCfg
    """Configuration for the visual servoing trajectory command generator."""

    def __init__(self, cfg: VsTrajCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the tag command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the pose of the base asset relative to the references (assumed to be fixed)
        self.ref_base_poses = [
            utils.get_prim_relative_pose(cfg.pose_ref_prim_name, ref=ref).to(device=self.device)
            for ref in cfg.ref_prim_names
        ]

        # create buffers to store the command
        # -- commands: ([pose], [twist])
        self.ref_ee_pose = torch.zeros((env.num_envs, 7), device=self.device)
        self.ee_twist = torch.zeros((env.num_envs, 6), device=self.device)

    def __str__(self) -> str:
        msg = "VsTrajCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> Tensor:
        """The current command (pose + twist) for each environment. Shape is (num_envs, 13)."""
        return torch.cat([self.ref_ee_pose, self.ee_twist], dim=-1)

    @property
    def target_state_command(self) -> Tensor:
        """The latest generated EE command (pose + twist). Shape is (num_envs, 13)."""
        return self._env.command_manager.get_command(self.cfg.traj_command_name)[:, 1:]

    @property
    def binary_command(self) -> Tensor:
        """The latest generated binary command. Shape is (num_envs, 1)."""
        return self._env.command_manager.get_command(self.cfg.binary_command_name)

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics based on the current state."""
        # no metrics to track currently
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command for the specified environments."""
        # force a command update to follow end-effector pose command
        self.time_left[env_ids] = torch.inf
        self._update_command()

    def _update_command(self) -> None:
        """Update the command based on the current state."""
        # get latest generated state command. Pose is (Base <- EE (target))
        base_ee_state = self.target_state_command
        base_ee_pose, self.ee_twist = base_ee_state[:, :7], base_ee_state[:, 7:]

        # get latest generated binary command and resolve (Ref <- Base)
        bin_cmd = self.binary_command.bool()
        ref_base_pose = torch.where(bin_cmd, self.ref_base_poses[1], self.ref_base_poses[0])

        # compute target pose relative to the chosen reference. Pose is (Ref <- EE (target))
        self.ref_ee_pose = utils.get_combined_pose(ref_base_pose, base_ee_pose)
