from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import warp as wp
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from torch import Tensor

    from .commands_cfg import TrajSmCommandCfg

import eye_injection.tasks.utils.isaac.common as utils

# initialize warp
wp.init()


class TrajSmCommand(CommandTerm):
    """Trajectory FSM-based state command term generator.

    This implementation follows the PickAndLiftSm example using warp on the IssacLab GitHub page at:
    https://github.com/isaac-sim/IsaacLab/blob/main/scripts/environments/state_machine/lift_cube_sm.py

    The command generator evaluates the discrete state of the robot's trajectory and sets its
    desired pose according to the command's configuration. Each state command consists of three
    components; discrete state, desired pose and desired twist. The FSM consists of the following
    five states:

    1) Setup (moving to the pose to start approach from).
    2) Approach (moving linearly along the z-axis at a fixed speed).
    3) Hold (remain stationary at the target pose for a certain period of time).
    4) Retreat (moving linearly along the negative z-axis at a fixed speed).
    5) Done (remain stationary at the final pose until timeout).

    **Note:** All poses are assumed to be relative to the robot's root body and **not** the world
    frame.
    """

    cfg: TrajSmCommandCfg
    """Configuration for the discrete state command generator."""

    def __init__(self, cfg: TrajSmCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the tag command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.dt = env.step_dt
        mtn_cfg = cfg.motion_cfg

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # extract the target prim poses and create approach and target poses
        prim_poses = [
            utils.get_prim_relative_pose(tgt, ref=cfg.ref_prim_name).to(device=self.device)
            for tgt in cfg.target_prim_names
        ]
        self.apr_ee_poses = [
            utils.apply_delta_offset(p, mtn_cfg.approach_offset, (0.0, 0.0, -1.0))
            for p in prim_poses
        ]
        self.tgt_ee_poses = [
            utils.apply_delta_offset(p, mtn_cfg.target_offset, (0.0, 0.0, -1.0)) for p in prim_poses
        ]
        # convert all poses from (w, x, y, z) to (x, y, z, w) for compatibility with warp
        self.apr_ee_poses = [p[:, [0, 1, 2, 4, 5, 6, 3]] for p in self.apr_ee_poses]
        self.tgt_ee_poses = [p[:, [0, 1, 2, 4, 5, 6, 3]] for p in self.tgt_ee_poses]

        # calculate wait times for each state
        apr_time = (mtn_cfg.approach_offset - mtn_cfg.target_offset) / mtn_cfg.approach_vel
        hld_time = mtn_cfg.stationary_time
        rtr_time = (mtn_cfg.approach_offset - mtn_cfg.target_offset) / mtn_cfg.retreat_vel

        # initialize state machine
        self.sm_dt = torch.full((env.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((env.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((env.num_envs,), device=self.device)
        self.des_ee_pose = torch.zeros((env.num_envs, 7), device=self.device)
        self.apr_ee_pose = torch.zeros((env.num_envs, 7), device=self.device)
        self.tgt_ee_pose = torch.zeros((env.num_envs, 7), device=self.device)
        self.state_wait_time = torch.tensor(
            [0.0, apr_time, hld_time, rtr_time, 0.0], device=self.device
        )
        self.state_z_vel = torch.tensor(
            [0.0, mtn_cfg.approach_vel, 0.0, -mtn_cfg.retreat_vel, 0.0], device=self.device
        )
        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.apr_ee_pose_wp = wp.from_torch(self.apr_ee_pose, wp.transform)
        self.tgt_ee_pose_wp = wp.from_torch(self.tgt_ee_pose, wp.transform)
        self.state_wait_time_wp = wp.from_torch(self.state_wait_time, wp.float32)

        # metrics
        self.metrics["pose_error"] = torch.zeros(self.num_envs, 7, device=self.device)

    def __str__(self) -> str:
        msg = "TrajSmCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> Tensor:
        """The current command (state + pose + twist) of each environment. Shape is (num_envs, 14)."""
        # combine state + pose + velocity
        state = self.sm_state[:, None]
        pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        twist = torch.zeros((self.num_envs, 6), device=self.device)
        twist[:, 2] = self.state_z_vel[self.sm_state]
        return torch.cat([state, pose, twist], dim=-1)

    @property
    def binary_command(self) -> Tensor:
        """The latest generated binary command. Shape is (num_envs, 1)."""
        return self._env.command_manager.get_command(self.cfg.binary_command_name)

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics based on the current state."""
        # compute pose error between robot and command poses
        ee_pose = utils.get_subtracted_pose(
            self.robot.data.root_pose_w, self.robot.data.body_pose_w[:, self.body_idx]
        )
        self.metrics["pose_error"] = utils.get_error_pose(
            ee_pose, self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        )

    def _resample_command(self, env_ids: Sequence[int] = None) -> None:
        """Resample the command for the specified environments."""
        if env_ids is None:
            env_ids = slice(None)
        # reset state machine
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0
        self.time_left[env_ids] = torch.inf
        # update poses according to binary command
        bin_cmd = self.binary_command.bool()
        self.apr_ee_pose = torch.where(bin_cmd, self.apr_ee_poses[1], self.apr_ee_poses[0])
        self.tgt_ee_pose = torch.where(bin_cmd, self.tgt_ee_poses[1], self.tgt_ee_poses[0])
        self.apr_ee_pose_wp = wp.from_torch(self.apr_ee_pose, wp.transform)
        self.tgt_ee_pose_wp = wp.from_torch(self.tgt_ee_pose, wp.transform)

    def _update_command(self) -> None:
        """Update the command based on the current state."""
        # get robot target body pose relative to its root body
        ee_pose = utils.get_subtracted_pose(
            self.robot.data.root_pose_w, self.robot.data.body_pose_w[:, self.body_idx]
        )[:, [0, 1, 2, 4, 5, 6, 3]]
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                self.des_ee_pose_wp,
                self.apr_ee_pose_wp,
                self.tgt_ee_pose_wp,
                self.state_wait_time_wp,
                self.cfg.motion_cfg.pose_tol[0],
                self.cfg.motion_cfg.pose_tol[1],
            ],
            device=self.device,
        )


"""
Warp SM implementation
"""


class TrajSmState:
    """States for the trajectory state machine."""

    Setup = wp.constant(0)
    Approach = wp.constant(1)
    Hold = wp.constant(2)
    Retreat = wp.constant(3)
    Done = wp.constant(4)


@wp.func
def is_pose_converged(cur: wp.transform, des: wp.transform, ttol: float, rtol: float) -> bool:  # type: ignore
    """Evaluate whether the pose has converged according to given tolerances."""
    p_cur = wp.transform_get_translation(cur)
    p_des = wp.transform_get_translation(des)
    if wp.length(p_cur - p_des) >= ttol:
        return False
    q_cur = wp.transform_get_rotation(cur)
    q_des = wp.transform_get_rotation(des)
    dot = wp.dot(q_cur, q_des)
    abs_dot = wp.clamp(wp.abs(dot), 0.0, 1.0)
    angle = 2.0 * wp.acos(abs_dot)
    return angle < rtol


@wp.func
def interp_pos(init: wp.transform, final: wp.transform, step: float) -> wp.transform:  # type: ignore
    """Interpolate between the positions of the two input poses according to given step size [0, 1]."""
    p_init = wp.transform_get_translation(init)
    p_final = wp.transform_get_translation(final)
    p_interp = p_init * (1.0 - step) + p_final * step
    return wp.transform(p_interp, wp.transform_get_rotation(init))


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),  # type: ignore
    sm_state: wp.array(dtype=int),  # type: ignore
    sm_wait_time: wp.array(dtype=float),  # type: ignore
    ee_pose: wp.array(dtype=wp.transform),  # type: ignore
    des_ee_pose: wp.array(dtype=wp.transform),  # type: ignore
    apr_ee_pose: wp.array(dtype=wp.transform),  # type: ignore
    tgt_ee_pose: wp.array(dtype=wp.transform),  # type: ignore
    state_wait_time: wp.array(dtype=float),  # type: ignore
    ttol: float,
    rtol: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == TrajSmState.Setup:
        des_ee_pose[tid] = apr_ee_pose[tid]
        # check for pose convergence
        if is_pose_converged(ee_pose[tid], des_ee_pose[tid], ttol, rtol):
            # move to next state and reset wait time
            sm_state[tid] = TrajSmState.Approach
            sm_wait_time[tid] = 0.0
    elif state == TrajSmState.Approach:
        des_ee_pose[tid] = interp_pos(
            apr_ee_pose[tid], tgt_ee_pose[tid], sm_wait_time[tid] / state_wait_time[state]
        )
        # wait for set time
        if sm_wait_time[tid] >= state_wait_time[state]:
            # move to next state and reset wait time
            sm_state[tid] = TrajSmState.Hold
            sm_wait_time[tid] = 0.0
    elif state == TrajSmState.Hold:
        des_ee_pose[tid] = tgt_ee_pose[tid]
        # wait for set time
        if sm_wait_time[tid] >= state_wait_time[state]:
            # move to next state and reset wait time
            sm_state[tid] = TrajSmState.Retreat
            sm_wait_time[tid] = 0.0
    elif state == TrajSmState.Retreat:
        des_ee_pose[tid] = interp_pos(
            tgt_ee_pose[tid], apr_ee_pose[tid], sm_wait_time[tid] / state_wait_time[state]
        )
        # wait for set time
        if sm_wait_time[tid] >= state_wait_time[state]:
            # move to next state and reset wait time
            sm_state[tid] = TrajSmState.Done
            sm_wait_time[tid] = 0.0
    elif state == TrajSmState.Done:
        des_ee_pose[tid] = apr_ee_pose[tid]
        # wait indefinitely
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
