# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer
    from torch import Tensor


class command_error_staged(ManagerTermBase):
    """Compute weighted command error based on the current stage.

    The function computes the command error between the published pose command and the current pose
    using a specified error function. The error is then weighted based on the current stage of the
    command. Stages refer to the discrete phases of the command motion execution (see PoseCommand
    docs for details).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        assert "ft_asset_name" in cfg.params, "FrameTransformer asset name must be specified."
        assert "command_name" in cfg.params, "PoseCommand name must be specified."
        assert "stage_weights" in cfg.params, "Stage weights must be specified."
        assert "error_fn" in cfg.params, "Error function must be specified."

        # find and store the stage weights as tensor
        self.stage_weights = torch.tensor(cfg.params.get("stage_weights"), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ft_asset_name: str,
        command_name: str,
        stage_weights: Sequence[float],
        error_fn: Callable[[Tensor, FrameTransformer, dict[str, Any]], Tensor],
        error_fn_kwargs: dict[str, Any] = {},
    ) -> Tensor:
        """Compute weighted command error based on the current stage.

        Args:
            env: The current active environment.
            ft_asset_name: FrameTransformer asset name for retreiving the current pose.
            command_name: PoseCommand name for retreiving the latest command.
            stage_weights: Weights for each command stage. Shape is (S,).
            error_fn: Function for computing the command error between the desired and current
                poses given the command and frame transformer asset.
            error_fn_kwargs: Additional keyword arguments for the error function.

        Returns:
            The weighted command error based on the current stage. Shape is (N,).
        """
        # extract the asset and command
        asset: FrameTransformer = env.scene[ft_asset_name]
        command = env.command_manager.get_command(command_name)

        # compute the weighted errors
        errors = error_fn(command, asset, **error_fn_kwargs)
        weights = self.stage_weights[command[:, 0].long()]
        return errors * weights
