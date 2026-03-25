# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import RigidObjectCollectionCfg
from isaaclab.utils import configclass

from eye_injection.tasks.utils.isaac.room_cfg import ROOM_CFG

from .env_apriltags_cfg import EyeInjectionEnvAprilTagsCfg, EyeInjectionSceneAprilTagsCfg

##
# Scene definition
##


@configclass
class EyeInjectionSceneEnclosedCfg(EyeInjectionSceneAprilTagsCfg):
    """Extended configuration for enclosed room scene."""

    # simple room
    ground = ROOM_CFG["Ground"].replace(prim_path="{ENV_REGEX_NS}/Ground")
    walls = RigidObjectCollectionCfg(
        rigid_objects={
            k: v.replace(prim_path="{ENV_REGEX_NS}" + f"/{k}")
            for k, v in ROOM_CFG.items()
            if "Wall" in k
        }
    )


##
# Environment configuration
##


@configclass
class EyeInjectionEnvEnclosedCfg(EyeInjectionEnvAprilTagsCfg):
    # Scene settings
    scene: EyeInjectionSceneEnclosedCfg = EyeInjectionSceneEnclosedCfg(
        num_envs=4096, env_spacing=6.5, replicate_physics=False
    )
