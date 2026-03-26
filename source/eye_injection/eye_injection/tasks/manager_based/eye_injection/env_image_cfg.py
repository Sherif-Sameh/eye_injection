# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from eye_injection.tasks.utils.isaac.room_cfg import GROUND_TEXTURE_PATHS

from . import mdp
from .env_base_cfg import (
    EventBaseCfg,
    EyeInjectionEnvBaseCfg,
    EyeInjectionSceneBaseCfg,
    ObservationsBaseCfg,
)

##
# Scene definition
##


@configclass
class EyeInjectionSceneImageCfg(EyeInjectionSceneBaseCfg):
    """Extended configuration for image-based scene."""

    # robot wrist camera
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link/flange/tool0/Camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.04, 0.135), rot=(0.9884, -0.1521, 0.0, 0.0), convention="ros"
        ),
    )


##
# MDP settings
##


@configclass
class ObservationsImageCfg(ObservationsBaseCfg):
    """Extended observation specifications for the image-based MDP."""

    @configclass
    class PolicyExtrCfg(ObsGroup):
        """Observations for policy extroceptive sensor group."""

        # observation terms (order preserved)
        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "normalize": False},
            history_length=0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy_extr: PolicyExtrCfg = PolicyExtrCfg()


@configclass
class EventImageCfg(EventBaseCfg):
    """Extended configuration for events for the image-based MDP."""

    rand_ground_texture = EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ground"),
            "event_name": "rand_ground_texture",
            "texture_paths": GROUND_TEXTURE_PATHS,
            "texture_rotation": (0.0, 6.283185307179586),
        },
    )


##
# Environment configuration
##


@configclass
class EyeInjectionEnvImageCfg(EyeInjectionEnvBaseCfg):
    # Scene settings
    scene: EyeInjectionSceneImageCfg = EyeInjectionSceneImageCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsImageCfg = ObservationsImageCfg()
    # MDP settings
    events: EventImageCfg = EventImageCfg()
