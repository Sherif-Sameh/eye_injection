# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from . import mdp
from .env_base_cfg import CommandsBaseCfg
from .env_enclosed_cfg import EyeInjectionEnvEnclosedCfg, EyeInjectionSceneEnclosedCfg
from .env_image_cfg import ObservationsImageCfg

##
# Scene definition
##


@configclass
class EyeInjectionSceneVsCfg(EyeInjectionSceneEnclosedCfg):
    """Extended configuration for visual servoing scene."""

    # no changes to assets

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

    # Tag pose command for retargeting pose commands to AprilTags
    tag_pose = mdp.TagPoseCommandCfg(
        camera_asset_name="camera",
        tag_prim_names=[
            "/World/envs/env_.*/Bed/Marker_1",
            "/World/envs/env_.*/Bed/Marker_2",
        ],
        tag_ids=[0, 1],
        pose_source_prim_name="/World/envs/env_.*/Robot/base_link",
        pose_command_name="target_pose",
        debug_vis=False,
    )


@configclass
class ActionsVsCfg:
    """Action specifications for the visual servoing MDP."""

    # Joint velocity (unscaled) action configuration
    # Joint velocity limits are extracted from the UR10e's technical specs sheet
    joint_vel = mdp.JointVelocityActionCfg(
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
        tag_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "tag_pose"}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups (override policy commands group)
    policy_cmd: PolicyCmdVsCfg = PolicyCmdVsCfg()


##
# Environment configuration
##


@configclass
class EyeInjectionEnvVsCfg(EyeInjectionEnvEnclosedCfg):
    # Scene settings
    scene: EyeInjectionSceneVsCfg = EyeInjectionSceneVsCfg(
        num_envs=1, env_spacing=5.0, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsVsCfg = ObservationsVsCfg()
    actions: ActionsVsCfg = ActionsVsCfg()
    commands: CommandsVsCfg = CommandsVsCfg()
