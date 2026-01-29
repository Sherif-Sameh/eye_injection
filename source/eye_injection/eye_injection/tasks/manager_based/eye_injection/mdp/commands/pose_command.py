# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_apply,
    quat_unique,
)

if TYPE_CHECKING:
    from torch import Tensor, BoolTensor
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import PoseCommandCfg


class PoseCommand(CommandTerm):
    """Pose command term for generating desired motions around target assets.

    The command generator generates poses according to the classical motion pattern of
    approach -> move linearly -> remain stationary -> retreat linearly -> done. This motion
    pattern's configuration is set through the PoseCommandCfg.MotionCfg class.

    In addition to the pose commands, the generator publishes the current discrete state
    of the trajectory denoting the active state of the five states described above.

    The command generator is configured to work alongside the BinaryCommand generator.
    Therefore, two target assests are defined and the choice of the target is set through
    the generated binary command.
    """

    cfg: PoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: PoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # extract the target assets' relative poses and approach vectors (assumed to be fixed)
        self.pose_target = self._get_relative_target_poses()
        self.vec_approach = self._get_approach_vectors(*self.pose_target)

        # offset the target poses along the negative direction of the approach vectors
        self.pose_target[0][:, :3] -= (
            self.vec_approach[0] * self.cfg.motion_cfg.target_offset
        )
        self.pose_target[1][:, :3] -= (
            self.vec_approach[1] * self.cfg.motion_cfg.target_offset
        )

        # create buffers for storing the current pose target and approach vector
        self.pose_target_curr = torch.zeros_like(self.pose_target[0])
        self.vec_approach_curr = torch.zeros_like(self.vec_approach[0])

        # compute number of steps for approach, stationary and retreat phases
        self.dt = env.step_dt
        self.n_steps_approach = math.floor(
            (self.cfg.motion_cfg.approach_offset / self.cfg.motion_cfg.approach_vel)
            / env.step_dt
        )
        self.n_steps_stationary = math.ceil(
            self.cfg.motion_cfg.stationary_time / env.step_dt
        )
        self.n_steps_retreat = math.floor(
            (self.cfg.motion_cfg.approach_offset / self.cfg.motion_cfg.retreat_vel)
            / env.step_dt
        )

        # create buffers to store the command
        # -- commands: (state, x, y, z, qw, qx, qy, qz) in root frame
        self.state = torch.zeros(self.num_envs, device=self.device)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        msg = "PoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> Tensor:
        """The current discrete state and desired pose in base frame. Shape is (num_envs, 8)."""
        return torch.cat([self.state[:, None], self.pose_command_b], dim=-1)

    @property
    def binary_command(self) -> Tensor:
        """The latest generated binary command. Shape is (num_envs, 1)."""
        return self._env.command_manager.get_command(self.cfg.binary_command_name)

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics based on the current state.
        
        Computes the pose error between the current robot pose and the commanded pose. Pose errors
        are measured and logged through two metrics. Euclidean distance (in m) is used for
        positioning errors and geodesic distance (in rad) is used for orientation errors.
        """
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = (
            combine_frame_transforms(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                self.pose_command_b[:, :3],
                self.pose_command_b[:, 3:],
            )
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command for the specified environments.
        
        Reset the discrete state of given environments and resamples the first pose command. The
        pose command is set to one of the two approach poses of the associated targets. The choice
        of target pose is determined through the binary command of each environment.
        """
        # reset discrete state to approach state
        self.state[env_ids] = 0
        self.time_left[env_ids] = torch.inf

        # sample target pose and approach vector according to the generated binary command
        binary_command = self.binary_command[env_ids].bool()
        self.pose_target_curr[env_ids] = torch.where(
            binary_command, self.pose_target[1][env_ids], self.pose_target[0][env_ids]
        )
        self.vec_approach_curr[env_ids] = torch.where(
            binary_command, self.vec_approach[1][env_ids], self.vec_approach[0][env_ids]
        )

        # update target pose
        self.pose_command_b[env_ids] = self.pose_target_curr[env_ids]
        self.pose_command_b[env_ids, :3] -= (
            self.vec_approach_curr[env_ids] * self.cfg.motion_cfg.approach_offset
        )

    def _update_command(self) -> None:
        """Update the command based on the current state."""
        # 1) approach: transition to move linearly if approach pose has been reached
        approach_done = torch.logical_and(self.state == 0, self._is_pose_reached())
        self.state[approach_done] = 1
        self.command_counter[approach_done] = 0

        # 2) move linearly: update commanded pose at approach velocity for n_steps_approach 
        move_active = self.state == 1
        if move_active.any():
            move_continue = torch.logical_and(
                move_active, self.command_counter < self.n_steps_approach
            )
            move_done = torch.logical_and(move_active, ~move_continue)

            self.pose_command_b[move_continue, :3] += (
                self.vec_approach_curr[move_continue]
                * self.cfg.motion_cfg.approach_vel
                * self.dt
            )
            self.command_counter[move_continue] += 1
            self.pose_command_b[move_done] = self.pose_target_curr[move_done]
            self.state[move_done] = 2
            self.command_counter[move_done] = 0

        # 3) remain stationary: transition to retreat if stationary time has elapsed
        stationary_active = self.state == 2
        if stationary_active.any():
            stationary_continue = torch.logical_and(
                stationary_active, self.command_counter < self.n_steps_stationary
            )
            stationary_done = torch.logical_and(stationary_active, ~stationary_continue)

            self.command_counter[stationary_continue] += 1
            self.state[stationary_done] = 3
            self.command_counter[stationary_done] = 0

        # 4) retreat: update commanded pose at retreat velocity for n_steps_retreat
        retreat_active = self.state == 3
        if retreat_active.any():
            retreat_continue = torch.logical_and(
                retreat_active, self.command_counter < self.n_steps_retreat
            )
            retreat_done = torch.logical_and(retreat_active, ~retreat_continue)

            self.pose_command_b[retreat_continue, :3] -= (
                self.vec_approach_curr[retreat_continue]
                * self.cfg.motion_cfg.retreat_vel
                * self.dt
            )
            self.command_counter[retreat_continue] += 1
            self.pose_command_b[retreat_done, :3] = (
                self.pose_target_curr[retreat_done, :3]
                - self.vec_approach_curr[retreat_done]
                * self.cfg.motion_cfg.approach_offset
            )
            self.state[retreat_done] = 4
            self.command_counter[retreat_done] = 0

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage

        Args:
            debug_vis: Whether to display visualization objects for commands or not.
        """
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:]
        )
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(
            body_link_pose_w[:, :3], body_link_pose_w[:, 3:]
        )

    """
    Private helper functions.
    """

    def _get_relative_target_poses(self) -> tuple[Tensor, Tensor]:
        """Get the relative poses of the target assets with respect to the body asset.

        Returns:
            A tuple containing the poses of the two target assets with respect to the base asset.
        """
        # extract the target and reference prims
        target_prims = (
            sim_utils.find_matching_prims(self.cfg.target_prim_names[0]),
            sim_utils.find_matching_prims(self.cfg.target_prim_names[1]),
        )
        ref_prims = sim_utils.find_matching_prims(self.cfg.source_prim_name)
        assert len(target_prims[0]) == len(target_prims[1]), (
            f"Target prims must have matching lengths, got {len(target_prims[0])} and {len(target_prims[1])}."
        )
        assert len(target_prims[0]) == len(ref_prims), (
            f"Target and reference prims must have matching lengths, got {len(target_prims[0])} and {len(ref_prims)}."
        )

        # extract the relative poses of the target prims
        pos_1, quat_1, pos_2, quat_2 = [], [], [], []
        for t_prim_1, t_prim_2, r_prim in zip(
            target_prims[0], target_prims[1], ref_prims
        ):
            # first target
            pos, quat = sim_utils.resolve_prim_pose(t_prim_1, ref_prim=r_prim)
            pos_1.append(torch.tensor(pos))
            quat_1.append(torch.tensor(quat))

            # second target
            pos, quat = sim_utils.resolve_prim_pose(t_prim_2, ref_prim=r_prim)
            pos_2.append(torch.tensor(pos))
            quat_2.append(torch.tensor(quat))

        # stack poses across the env dim and move to device
        pos_1 = torch.stack(pos_1, dim=0).to(device=self.device)
        quat_1 = torch.stack(quat_1, dim=0).to(device=self.device)
        pos_2 = torch.stack(pos_2, dim=0).to(device=self.device)
        quat_2 = torch.stack(quat_2, dim=0).to(device=self.device)

        # make quaternions unique (+ve real part) if required
        if self.cfg.make_quat_unique:
            quat_1 = quat_unique(quat_1)
            quat_2 = quat_unique(quat_2)

        pose_1 = torch.cat([pos_1, quat_1], dim=-1)
        pose_2 = torch.cat([pos_2, quat_2], dim=-1)
        return pose_1, pose_2

    def _get_approach_vectors(
        self, pose_1: Tensor, pose_2: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Get the approach trajectories according to the target poses and motion configuration.

        Args:
            pose_1: The first target pose. Shape is (N, 7).
            pose_2: The second target pose. Shape is (N, 7).

        Returns:
            A tuple containing the approach trajectories of the two target assets with respect to
                the base asset. Shape of each entry is (N, M, 3).
        """
        # create the approach vector along the Z-axis of the target frame
        approach = torch.zeros(self.num_envs, 3, device=self.device)
        approach[:, 2] = 1

        # rotate the approach vector by the current orientations
        approach_1 = quat_apply(pose_1[:, 3:], approach)
        approach_2 = quat_apply(pose_2[:, 3:], approach)
        return approach_1, approach_2

    def _is_pose_reached(self) -> BoolTensor:
        """Compute whether the target pose has been reached according to set tolerances.

        Returns:
            Boolean tensor containing the success status of each entry according to its errors and
                the set tolerances. Shape is (N,).
        """
        reached = torch.logical_and(
            self.metrics["position_error"] <= self.cfg.motion_cfg.pose_tol[0],
            self.metrics["orientation_error"] <= self.cfg.motion_cfg.pose_tol[1],
        )
        return reached
