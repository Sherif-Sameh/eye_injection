# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from eye_injection.tasks.utils.isaac import PhysxReducedCfg
from eye_injection.tasks.utils.isaac.room_cfg import ROOM_CFG, ROOM_THICKNESS

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets import UR10e_CFG  # isort:skip

##
# Scene definition
##


@configclass
class EyeInjectionSceneBaseCfg(InteractiveSceneCfg):
    """Configuration for the base manipulation eye-injection scene."""

    # ground
    ground = ROOM_CFG["Ground"].replace(
        prim_path="{ENV_REGEX_NS}/Ground",
        spawn=ROOM_CFG["Ground"].spawn.replace(size=(3.0, 3.0, ROOM_THICKNESS)),
    )

    # robot
    robot: ArticulationCfg = UR10e_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UR10e_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=5.0
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.6272,
                "shoulder_lift_joint": -1.2816,
                "elbow_joint": 1.2922,
                "wrist_1_joint": -1.5829,
                "wrist_2_joint": -1.5709,
                "wrist_3_joint": 0.6273,
            },
            pos=(0.15, -0.7, 1.05),
        ),
    )

    # robot stand
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Stand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    # frame transformer for EE w.r.t. robot's base link
    frame_ee = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link", name="end_effector"
            )
        ],
        debug_vis=False,
    )

    # contact sensor (for collision checking)
    contact_forces_robot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(?!base_link).*_link",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )

    # examination bed
    bed = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/ExaminationBed.usd")
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(0.7071, 0.0, 0.0, -0.7071)
        ),
    )

    # human
    person = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed/Person",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/Person.usd"), scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.9, 0.975), rot=(0.70711, -0.70711, 0.0, 0.0)
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )

    def __post_init__(self):
        # Disable PD joint position tracking to allow for pure joint torque control
        for key in self.robot.actuators:
            self.robot.actuators[key].stiffness = 0.0
            self.robot.actuators[key].damping = 0.0


##
# MDP settings
##


@configclass
class CommandsBaseCfg:
    """Command terms for the base MDP."""

    # Binary command corresponding to the targeted eye (0 = left, 1 = right)
    target_eye = mdp.BinaryCommandCfg(prob_1=0.5)

    # Trajectory SM command (not observable to agent)
    target_traj = mdp.TrajSmCommandCfg(
        asset_name="robot",
        body_name="wrist_3_link",
        target_prim_names=(
            "/World/envs/env_.*/Bed/Person/Person/Root/EyeLeft",
            "/World/envs/env_.*/Bed/Person/Person/Root/EyeRight",
        ),
        ref_prim_name="/World/envs/env_.*/Robot/base_link",
        binary_command_name="target_eye",
        motion_cfg=mdp.TrajSmCommandCfg.MotionCfg(
            pose_tol=(0.1, 0.1),
            target_offset=0.1,
            approach_offset=0.3,
            approach_vel=0.05,
            stationary_time=4.0,
            retreat_vel=0.05,
        ),
    )


@configclass
class ActionsBaseCfg:
    """Action specifications for the base MDP."""

    # Joint torque action configuration
    # Joint limits are extracted from the UR10e's URDF file
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        clip={
            "shoulder_pan_joint": (-330.0, 330.0),
            "shoulder_lift_joint": (-330.0, 330.0),
            "elbow_joint": (-150.0, 150.0),
            "wrist_1_joint": (-56.0, 56.0),
            "wrist_2_joint": (-56.0, 56.0),
            "wrist_3_joint": (-56.0, 56.0),
        },
        scale={
            "shoulder_pan_joint": 330.0,
            "shoulder_lift_joint": 330.0,
            "elbow_joint": 150.0,
            "wrist_1_joint": 56.0,
            "wrist_2_joint": 56.0,
            "wrist_3_joint": 56.0,
        },
    )


@configclass
class ObservationsBaseCfg:
    """Observation specifications for the base MDP."""

    @configclass
    class PolicyCmdCfg(ObsGroup):
        """Observations for policy commands group."""

        # observation terms (order preserved)
        target_eye_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_eye"}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyPropCfg(ObsGroup):
        """Observations for policy proprioceptive sensor group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy_cmd: PolicyCmdCfg = PolicyCmdCfg()
    policy_prop: PolicyPropCfg = PolicyPropCfg()


@configclass
class EventBaseCfg:
    """Configuration for events for the base MDP."""

    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.125, 0.125),
            "velocity_range": (0.0, 0.0),
        },
    )

    rand_robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )


@configclass
class RewardsBaseCfg:
    """Reward terms for the base MDP."""

    # (1) Position tracking reward
    position_tracking = RewTerm(
        func=mdp.command_error_staged,
        weight=-0.5,
        params={
            "ft_asset_name": "frame_ee",
            "command_name": "target_traj",
            "stage_weights": [0.1, 0.3, 0.8, 0.3, 0.1],
            "error_fn": mdp.position_command_error,
            "error_fn_kwargs": {},
        },
    )
    # (2) Orientation tracking reward
    orientation_tracking = RewTerm(
        func=mdp.command_error_staged,
        weight=-0.5,
        params={
            "ft_asset_name": "frame_ee",
            "command_name": "target_traj",
            "stage_weights": [0.1, 0.3, 0.8, 0.3, 0.1],
            "error_fn": mdp.orientation_command_error,
            "error_fn_kwargs": {},
        },
    )
    # (3) Action penalty
    action = RewTerm(func=mdp.action_l2, weight=-0.005)
    # (4) Action rate penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # (5) Joint velocity penalty
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # (6) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class TerminationsBaseCfg:
    """Termination terms for the base MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Collision Termination
    collision = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_robot"), "threshold": 0.1},
    )


##
# Environment configuration
##


@configclass
class EyeInjectionEnvBaseCfg(ManagerBasedRLEnvCfg):
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        physx=PhysxReducedCfg(has_soft_bodies=False, has_particles=False)
    )
    # Scene settings
    scene: EyeInjectionSceneBaseCfg = EyeInjectionSceneBaseCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsBaseCfg = ObservationsBaseCfg()
    actions: ActionsBaseCfg = ActionsBaseCfg()
    commands: CommandsBaseCfg = CommandsBaseCfg()
    # MDP settings
    rewards: RewardsBaseCfg = RewardsBaseCfg()
    terminations: TerminationsBaseCfg = TerminationsBaseCfg()
    events: EventBaseCfg = EventBaseCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15.0
        # viewer settings
        self.viewer.eye = (-1.0, 1.0, 4.0)
        self.viewer.lookat = (0.5, -0.5, 0.5)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
