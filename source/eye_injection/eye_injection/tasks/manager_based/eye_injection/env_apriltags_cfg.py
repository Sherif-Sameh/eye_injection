# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .env_image_cfg import EyeInjectionEnvImageCfg, EyeInjectionSceneImageCfg

##
# Scene definition
##


@configclass
class EyeInjectionSceneAprilTagsCfg(EyeInjectionSceneImageCfg):
    """Extended configuration for scene with AprilTags."""

    # AprilTags
    marker_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed/Marker_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/Plane.usd"),
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=str(Path(__file__).parent / "materials/AprilTag_00.mdl")
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.75, 0.794)),
    )
    marker_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed/Marker_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/Plane.usd"),
            scale=(0.1, 0.1, 1.0),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=str(Path(__file__).parent / "materials/AprilTag_01.mdl")
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, 0.75, 0.794)),
    )


##
# Environment configuration
##


@configclass
class EyeInjectionEnvAprilTagsCfg(EyeInjectionEnvImageCfg):
    # Scene settings
    scene: EyeInjectionSceneAprilTagsCfg = EyeInjectionSceneAprilTagsCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=False
    )
