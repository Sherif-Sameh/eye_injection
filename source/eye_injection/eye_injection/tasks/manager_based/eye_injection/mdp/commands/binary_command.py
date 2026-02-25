# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import BinaryCommandCfg


class BinaryCommand(CommandTerm):
    """Binary command term for uniformly generating binary goals between 0 and 1.

    Commands are only resampled at episode resets.
    """

    cfg: BinaryCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: BinaryCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers to store the command
        self._prob = torch.full((self.num_envs, 1), 0.5, device=self.device)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        # -- no metrics to track currently

    def __str__(self) -> str:
        msg = "BinaryCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired binary command. Shape is (num_envs, 1)."""
        return self._command

    """
    Operations.
    """

    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        # commands are not affected by the current state
        pass

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics based on the current state."""
        # no metrics to track currently
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command for the specified environments."""
        self._command[env_ids] = torch.bernoulli(self._prob[env_ids])

    def _update_command(self):
        """Update the command based on the current state."""
        # commands are not affected by the current state
        pass
