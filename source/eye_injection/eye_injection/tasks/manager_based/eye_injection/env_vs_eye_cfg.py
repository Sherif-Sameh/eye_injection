# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import mdp
from .env_image_cfg import ObservationsImageCfg
from .env_vs_cfg import EyeInjectionEnvVsCfg

##
# MDP settings
##


@configclass
class CommandsVsEyeCfg:
    """Command terms for visual servoing eye calibration MDP."""

    # keep only target eye command to avoid empty command vector
    target_eye = mdp.BinaryCommandCfg(prob_1=0.5)


##
# Environment configuration
##


@configclass
class EyeInjectionEnvVsEyeCfg(EyeInjectionEnvVsCfg):
    # Basic settings
    observations: ObservationsImageCfg = ObservationsImageCfg()
    commands: CommandsVsEyeCfg = CommandsVsEyeCfg()
