# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from eye_injection.tasks.utils.isaac import PhysxReducedCfg

from . import mdp
from .env_base_cfg import CommandsBaseCfg
from .env_image_cfg import EyeInjectionEnvImageCfg, EyeInjectionSceneImageCfg, ObservationsImageCfg

##
# Scene definition
##


@configclass
class EyeInjectionSceneVsCfg(EyeInjectionSceneImageCfg):
    """Extended configuration for visual servoing scene."""

    # AprilTags
    marker_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed/Marker_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/Plane.usd"),
            scale=(0.06, 0.06, 1.0),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=str(Path(__file__).parent / "materials/AprilTag_00.mdl")
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.145, 0.65, 0.794)),
    )
    marker_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed/Marker_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/Plane.usd"),
            scale=(0.06, 0.06, 1.0),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=str(Path(__file__).parent / "materials/AprilTag_01.mdl")
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.145, 0.65, 0.794)),
    )

    def __post_init__(self):
        # Disable joint stiffness to allow for pure joint velocity control
        for key in self.robot.actuators:
            self.robot.actuators[key].stiffness = 0.0


##
# MDP settings
##


@configclass
class CommandsVsCfg(CommandsBaseCfg):
    """Extended command terms for visual servoing MDP."""

    # VS trajectory command for retargeting state commands
    vs_traj = mdp.VsTrajCommandCfg(
        ref_prim_names=(
            "/World/envs/env_.*/Bed/Person/Person/Root/EyeLeft",
            "/World/envs/env_.*/Bed/Person/Person/Root/EyeRight",
        ),
        pose_ref_prim_name="/World/envs/env_.*/Robot/base_link",
        traj_command_name="target_traj",
        binary_command_name="target_eye",
        debug_vis=False,
    )


@configclass
class ActionsVsCfg:
    """Action specifications for the visual servoing MDP."""

    # Joint velocity (unscaled) action configuration
    # Joint velocity limits are extracted from the UR10e's technical specs sheet
    joint_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        clip={
            "shoulder_pan_joint": (-120 * math.pi / 180, 120 * math.pi / 180),
            "shoulder_lift_joint": (-120 * math.pi / 180, 120 * math.pi / 180),
            "elbow_joint": (-180 * math.pi / 180, 180 * math.pi / 180),
            "wrist_1_joint": (-180 * math.pi / 180, 180 * math.pi / 180),
            "wrist_2_joint": (-180 * math.pi / 180, 180 * math.pi / 180),
            "wrist_3_joint": (-180 * math.pi / 180, 180 * math.pi / 180),
        },
    )


@configclass
class ObservationsVsCfg(ObservationsImageCfg):
    """Modified observation specifications for the visual servoing MDP."""

    @configclass
    class PolicyCmdVsCfg(ObsGroup):
        """Modified observations for policy commands group."""

        # observation terms (order preserved)
        vs_traj_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "vs_traj"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups (override policy commands group)
    policy_cmd: PolicyCmdVsCfg = PolicyCmdVsCfg()


@configclass
class EventVsCfg:
    """Modfied configuration for events for the visual servoing MDP."""

    # keep only robot joint randomization
    # disable robot base, robot mass and ground texture randomization
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.125, 0.125),
            "velocity_range": (0.0, 0.0),
        },
    )

    # add gravity force randomization to simulate imperfect gravity compensation
    rand_gravity_force = EventTerm(
        func=mdp.apply_external_gravity_force,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_ids=range(1, 7)),
            "distribution_params": (-0.0, 0.0),
            "distribution": "uniform",
        },
    )


class RewardsVsCfg:
    """Reward terms for the visual servoing MDP."""

    # no reward terms are utilized for VS MDP


##
# Environment configuration
##


@configclass
class EyeInjectionEnvVsCfg(EyeInjectionEnvImageCfg):
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        physx=PhysxReducedCfg(
            partition_reduction=1, memory_reduction=4, has_soft_bodies=False, has_particles=False
        )
    )
    # Scene settings
    scene: EyeInjectionSceneVsCfg = EyeInjectionSceneVsCfg(
        num_envs=1, env_spacing=6.5, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsVsCfg = ObservationsVsCfg()
    actions: ActionsVsCfg = ActionsVsCfg()
    commands: CommandsVsCfg = CommandsVsCfg()
    # MDP settings
    rewards: RewardsVsCfg = RewardsVsCfg()
    events: EventVsCfg = EventVsCfg()
