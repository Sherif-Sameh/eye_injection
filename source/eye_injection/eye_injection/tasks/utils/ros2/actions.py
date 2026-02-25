from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor
    from trajectory_msgs.msg import JointTrajectory


def position_action(msg: JointTrajectory) -> Tensor:
    """Joint position control.

    Extract joint position actions from JointTrajectory message.

    Args:
        msg: JointTrajectory message sent by controller.

    Returns:
        Tensor containing the joint position actions. Shape is (num_points, num_joints).
    """
    action = np.array([pt.positions for pt in msg.points])
    return torch.from_numpy(action)


def position_action_velocity_ff(msg: JointTrajectory) -> Tensor:
    """Joint position control with feed-forward velocity inputs.

    Extract joint position and velocity actions from JointTrajectory message.

    Args:
        msg: JointTrajectory message sent by controller.

    Returns:
        Tensor containing the joint position and velocity actions. Shape is
            (num_points, num_joints * 2).
    """
    action = np.array([np.concatenate([pt.positions, pt.velocities], axis=-1) for pt in msg.points])
    return torch.from_numpy(action)


def position_action_effort_ff(msg: JointTrajectory) -> Tensor:
    """Joint position control with feed-forward effort inputs.

    Extract joint position and effort actions from JointTrajectory message.

    Args:
        msg: JointTrajectory message sent by controller.

    Returns:
        Tensor containing the joint position and effort actions. Shape is
            (num_points, num_joints * 2).
    """
    action = np.array([np.concatenate([pt.positions, pt.effort], axis=-1) for pt in msg.points])
    return torch.from_numpy(action)


def position_action_velocity_effort_ff(msg: JointTrajectory) -> Tensor:
    """Joint position control with feed-forward velocity and effort inputs.

    Extract joint position, velocity and effort actions from JointTrajectory message.

    Args:
        msg: JointTrajectory message sent by controller.

    Returns:
        Tensor containing the joint position, velocity and effort actions. Shape is
            (num_points, num_joints * 3).
    """
    action = np.array(
        [np.concatenate([pt.positions, pt.velocities, pt.effort], axis=-1) for pt in msg.points]
    )
    return torch.from_numpy(action)


def velocity_action(msg: JointTrajectory) -> Tensor:
    """Joint velocity control.

    Extract joint velocity actions from JointTrajectory message.

    Args:
        msg: JointTrajectory message sent by controller.

    Returns:
        Tensor containing the joint velocity actions. Shape is (num_points, num_joints).
    """
    action = np.array([pt.velocities for pt in msg.points])
    return torch.from_numpy(action)


def velocity_action_effort_ff(msg: JointTrajectory) -> Tensor:
    """Joint velocity control with feed-forward effort inputs.

    Extract joint velocity and effort actions from JointTrajectory message.

    Args:
        msg: JointTrajectory message sent by controller.

    Returns:
        Tensor containing the joint velocity and effort actions. Shape is
            (num_points, num_joints * 2).
    """
    action = np.array([np.concatenate([pt.velocities, pt.effort], axis=-1) for pt in msg.points])
    return torch.from_numpy(action)


def effort_action(msg: JointTrajectory) -> Tensor:
    """Joint effort control.

    Extract joint effort actions from JointTrajectory message.

    Args:
        msg: JointTrajectory message sent by controller.

    Returns:
        Tensor containing the joint effort actions. Shape is (num_points, num_joints).
    """
    action = np.array([pt.effort for pt in msg.points])
    return torch.from_numpy(action)
