# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .room_cfg import ROOM_CFG

##
# Pre-defined configs
##

from isaaclab_assets import UR10e_CFG  # isort:skip

##
# Scene definition
##


@configclass
class EyeInjectionSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # simple room
    room = RigidObjectCollectionCfg(
        rigid_objects={
            k: v.replace(prim_path="{ENV_REGEX_NS}" + f"/{k}") for k, v in ROOM_CFG.items()
        }
    )

    # examination bed
    bed = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/ExaminationBed.usd"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(0.7071, 0.0, 0.0, -0.7071),
        ),
    )

    # robot stand
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(1.4, 1.4, 2.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.15, 0.7, 1.05),
        ),
    )

    # robot
    robot: ArticulationCfg = UR10e_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UR10e_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": -1.5707963267948966 / 1.5,
                "shoulder_lift_joint": -1.5707963267948966 / 0.75,
                "elbow_joint": 1.5707963267948966,
                "wrist_1_joint": -1.5707963267948966,
                "wrist_2_joint": -1.5707963267948966,
                "wrist_3_joint": 0.0,
            },
            pos=(0.15, 0.7, 1.05),
        )
    )

    # wrist camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.06, 0.0, -0.035), rot=(1.0, 0.0, 0.0, 0.0), convention="ros",
        ),
    )

    # human
    person = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Person",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/male_adult_construction_03/male_adult_construction_03.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.9, 0.0, 0.95), rot=(0.5, -0.5, 0.5, -0.5),
        ),
    )

    # AprilTags
    marker_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Marker_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/Plane.usd"),
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=str(Path(__file__).parent / "materials/AprilTag_00.mdl"),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, 0.2, 0.794),
        ),
    )
    marker_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Marker_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/Plane.usd"),
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=str(Path(__file__).parent / "materials/AprilTag_01.mdl"),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.75, -0.2, 0.794),
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
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
class ActionsCfg:
    """Action specifications for the MDP."""
    
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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.125, 0.125),
            "velocity_range": (0.0, 0.0),
        },
    )

    # TODO: Add domain randomization for joint stiffness, damping and friction params


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Pose tracking reward
    # (2) Action penalty
    action = RewTerm(func=mdp.action_l2, weight=-0.005)
    # (3) Action rate penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # (4) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Collision Termination
    # collision = DoneTerm(
    #     func=mdp.undesired_contacts,
    #     params={"sensor_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "threshold": 4.0},
    # )


##
# Environment configuration
##


@configclass
class EyeInjectionEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: EyeInjectionSceneCfg = EyeInjectionSceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (-2.0, -2.0, 6.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation