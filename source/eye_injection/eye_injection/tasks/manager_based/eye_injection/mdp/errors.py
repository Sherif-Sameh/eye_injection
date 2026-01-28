# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.sensors import FrameTransformer
    from torch import Tensor


def position_command_error(command: Tensor, asset: FrameTransformer) -> Tensor:
    """Compute the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and
    the current position of the asset's body. The position error is computed as the L2-norm of the
    difference between the desired and current positions.

    Args:
        command: Tensor containing the latest generated PoseCommand. Shape is (N, 8).
        asset: FrameTransformer asset for retreiving the current pose with respect to the source
            frame. The frame transformer is expected to have the same source frame as the command
            generator.

    Returns:
        The position error using L2-norm between the current and desired positions. Shape is (N,).
    """
    # obtain the desired and current positions
    des_pos = command[:, 1:4]
    curr_pos = asset.data.target_pos_source[:, 0]
    return torch.norm(curr_pos - des_pos, dim=-1)


def position_command_error_tanh(
    command: Tensor,
    asset: FrameTransformer,
    std: float = 1.0,
) -> Tensor:
    """Compute the position error using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body and maps it to [0, 1] with a tanh kernel.

    Args:
        command: Tensor containing the latest generated PoseCommand. Shape is (N, 8).
        asset: FrameTransformer asset for retreiving the current pose with respect to the source
            frame. The frame transformer is expected to have the same source frame as the command
            generator.
        std: Standard deviation for dividing the distance before applying the tanh kernel.

    Returns:
        The position error using tanh kernel between the current and desired positions. Shape is (N,).
    """
    # obtain the desired and current positions
    des_pos = command[:, 1:4]
    curr_pos = asset.data.target_pos_source[:, 0]
    distance = torch.norm(curr_pos - des_pos, dim=-1)
    return torch.tanh(distance / std)


def orientation_command_error(command: Tensor, asset: FrameTransformer) -> Tensor:
    """Compute orientation error using shortest path (geodesic distance).

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body. The orientation error is computed as the shortest
    path between the desired and current orientations (geodesic distance).

    Args:
        command: Tensor containing the latest generated PoseCommand. Shape is (N, 8).
        asset: FrameTransformer asset for retreiving the current pose with respect to the source
            frame. The frame transformer is expected to have the same source frame as the command
            generator.

    Returns:
        The orientation error using the geodesic distance between the current and desired
            orientations. Shape is (N,).
    """
    # obtain the desired and current orientations
    des_quat = command[:, 4:]
    curr_quat = asset.data.target_quat_source[:, 0]
    return quat_error_magnitude(curr_quat, des_quat)
