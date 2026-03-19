# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from . import mdp
from .env_image_cfg import ObservationsImageCfg
from .env_vs_cfg import EyeInjectionEnvVsCfg, EyeInjectionSceneVsCfg

##
# Scene definition
##


@configclass
class EyeInjectionSceneVsCalibCfg(EyeInjectionSceneVsCfg):
    """Extended configuration for visual servoing calibration scene."""

    # replace person with calibration pattern
    person = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed/ChArUco",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "assets/Plane.usd"),
            scale=(0.3, 0.3, 1.0),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=str(Path(__file__).parent / "materials/charuco.mdl")
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.5, 0.794)),
    )


##
# MDP settings
##


@configclass
class CommandsVsCalibCfg:
    """Command terms for visual servoing calibration MDP."""

    # keep only target eye command to avoid empty command vector
    target_eye = mdp.BinaryCommandCfg(prob_1=0.5)


##
# Environment configuration
##


@configclass
class EyeInjectionEnvVsCalibCfg(EyeInjectionEnvVsCfg):
    # Scene settings
    scene: EyeInjectionSceneVsCalibCfg = EyeInjectionSceneVsCalibCfg(
        num_envs=1, env_spacing=5.0, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsImageCfg = ObservationsImageCfg()
    commands: CommandsVsCalibCfg = CommandsVsCalibCfg()
