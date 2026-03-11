from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.sensors import Camera
    from torch import BoolTensor, Tensor


def get_prim_relative_pose(
    target: str, ref: str | None = None, make_quat_unique: bool = False
) -> Tensor:
    """Get the pose of the target primitive relative to another reference primitive from IsaacSim.

    Args:
        target: Name of the target primitive. Can make use of any regex expressions supported
            within IsaacSim.
        ref: Optional name of the reference primitive. Can make use of any regex expressions
            supported within IsaacSim. If None, the pose is retrieved relative to the world
            primitive. Defaults to None.
        make_quat_unique: Make quaternions unique (i.e., have a +ve real part). Default to False.

    Returns:
        Relative pose of the target primitive wrt to the reference primitive. Shape is (n_prims, 7).
        Each pose is ordered as (tx, ty, tz, qw, qx, qy, qz).
    """
    # extract the target and reference prims
    target_prims = sim_utils.find_matching_prims(target)
    if ref is not None:
        ref_prims = sim_utils.find_matching_prims(ref)
        assert len(target_prims) == len(ref_prims), (
            "If reference prims are given, target and reference prims must have matching lengths,"
            f" got {len(target_prims)} and {len(ref_prims)}."
        )
    else:
        ref_prims = [None] * len(target_prims)
    # extract relative poses
    poses = [
        sim_utils.resolve_prim_pose(t_prim, ref_prim=r_prim)
        for t_prim, r_prim in zip(target_prims, ref_prims)
    ]
    pos = torch.stack([torch.tensor(p[0]) for p in poses], dim=0)
    quat = torch.stack([torch.tensor(p[1]) for p in poses], dim=0)
    quat = math_utils.quat_unique(quat) if make_quat_unique else quat
    return torch.cat([pos, quat], dim=-1)


def get_camera_relative_pose(
    camera: Camera, convention: Literal["opengl", "ros", "world"] = "ros"
) -> Tensor:
    """Get the camera pose relative to its parent primitive in given convention.

    Args:
        camera: Camera asset.
        convention: Target camera frame orientation convention.

    Returns:
        Relative pose of the camera in the given convention. Shape is (7,).
    """
    offset = camera.cfg.offset
    pos = torch.tensor(offset.pos)
    rot = torch.tensor(offset.rot)
    if offset.convention != convention:
        rot = math_utils.convert_camera_frame_orientation_convention(
            rot, origin=offset.convention, target=convention
        )
    return torch.cat([pos, rot], dim=-1)


@torch.jit.script
def get_eye_like(ref: Tensor) -> Tensor:
    """Get identity pose with the same shape as reference pose."""
    eye = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=ref.dtype, device=ref.device)
    return eye.expand_as(ref)


@torch.jit.script
def get_offset_pose(ref: Tensor, mag: float, dir: tuple[float, float, float]) -> Tensor:
    """Get an offset pose from reference along a certain direction by some magnitude.

    Args:
        ref: Reference pose for generating offset pose. Shape is (..., 7).
        mag: Magnitude of the offset in meters. Must be > 0.
        dir: Direction vector (x, y, z) for offset pose. Defined in the frame of the reference
            pose.

    Returns:
        Offset pose from reference along the given direction by the given magnitude.
        Shape is (..., 7).
    """
    assert mag > 0, f"Offset magnitude must be > 0, got {mag}."
    # Create dir vector
    dir = torch.tensor(dir, device=ref.device)
    dir /= torch.linalg.vector_norm(dir) + 1e-8
    dir = dir.expand(ref.shape[:-1] + (-1,))
    # Rotate and apply offset vector
    dir = math_utils.quat_apply(ref[..., 3:], dir)
    pose = torch.clone(ref)
    pose[..., :3] += dir * mag
    return pose


@torch.jit.script
def get_combined_pose(pose01: Tensor, pose12: Tensor) -> Tensor:
    """Get pose generated from combining the two given poses.

    Args:
        pose01: Pose of frame 1 wrt frame 0. Shape is (..., 7).
        pose12: Pose of frame 2 wrt frame 2. Shape is (..., 7).

    Returns:
        Combined pose of frame 2 wrt frame 0. Shape is (..., 7).
    """
    t01, q01 = pose01[..., :3], pose01[..., 3:]
    t12, q12 = pose12[..., :3], pose12[..., 3:]
    t02, q02 = math_utils.combine_frame_transforms(t01, q01, t12, q12)
    return torch.cat([t02, q02], dim=-1)


@torch.jit.script
def get_subtracted_pose(pose01: Tensor, pose02: Tensor) -> Tensor:
    """Get the pose generated from subtracting the two given poses.

    Args:
        pose01: Pose of frame 1 wrt frame 0. Shape is (..., 7).
        pose02: Pose of frame 2 wrt frame 0. Shape is (..., 7).

    Returns:
        Subtracted pose of frame 2 wrt frame 1. Shape is (..., 7).
    """
    t01, q01 = pose01[..., :3], pose01[..., 3:]
    t02, q02 = pose02[..., :3], pose02[..., 3:]
    t12, q12 = math_utils.subtract_frame_transforms(t01, q01, t02, q02)
    return torch.cat([t12, q12], dim=-1)


def get_interpolated_pose(first: Tensor, second: Tensor, step: float) -> Tensor:
    """Get the interpolated pose between two poses according to given step size.

    Args:
        first: First pose for interpolation. Shape is (..., 7).
        second: Second pose for interpolation. Shape is (..., 7).
        step: Step size between [0, 1] for interpolation.

    Returns:
        Interpolated pose. If step = 0, output = `first`. If step = 1, output = `second`.
        Shape is (..., 7).
    """
    if step == 0.0:
        return first
    if step == 1.0:
        return second
    t1, q1 = first[..., :3], first[..., 3:]
    t2, q2 = second[..., :3], second[..., 3:]
    # interpolate positions
    ti = t1 + (t2 - t1) * step
    # interpolate quaternions
    qi = math_utils.quat_slerp(q1, q2, step)
    return torch.cat([ti, qi], dim=-1)


@torch.jit.script
def apply_delta_pos(ref: Tensor, delta_pos: Tensor) -> Tensor:
    """Apply a delta position to a given reference pose.

    Args:
        ref: Reference pose to apply delta to. Shape is (..., 7).
        delta_pos: Delta position to apply to reference pose. Shape is (..., 3).

    Returns:
        Pose after applying delta positon. Shape is (..., 7).
    """
    td = ref[..., :3] + delta_pos
    return torch.cat([td, ref[..., 3:]], dim=-1)


@torch.jit.script
def apply_delta_rot(ref: Tensor, delta_rot: Tensor) -> Tensor:
    """Apply a delta rotation to a given reference pose.

    Args:
        ref: Reference pose to apply delta to. Shape is (..., 7).
        delta_rot: Delta rotation to apply to reference pose. Shape is (..., 3).

    Returns:
        Pose after applying delta rotation. Shape is (..., 7).
    """
    qd = math_utils.quat_box_plus(ref[..., 3:], delta_rot)
    return torch.cat([ref[..., :3], qd], dim=-1)


@torch.jit.script
def apply_delta_pose(ref: Tensor, delta_pose: Tensor) -> Tensor:
    """Apply a delta pose to a given reference pose.

    Args:
        ref: Reference pose to apply delta to. Shape is (..., 7).
        delta_pose: Delta pose to apply to reference pose. Shape is (..., 6).

    Returns:
        Pose after applying delta pose. Shape is (..., 7).
    """
    td, qd = math_utils.apply_delta_pose(ref[..., :3], ref[..., 3:], delta_pose)
    return torch.cat([td, qd], dim=-1)


@torch.jit.script
def get_error_pose(source: Tensor, target: Tensor) -> Tensor:
    """Get the pose error between a pose and a reference pose.

    **Note:** Both poses must share the same reference frame.

    Args:
        source: Pose of source frame for measuring pose error. Shape is (..., 7).
        target: Pose of target frame for measuring pose error. Shape is (..., 7).

    Returns:
        Pose representing the error between the source and target poses in the common reference
        frame. Shape is (..., 7).
    """
    t01, q01 = source[..., :3], source[..., 3:]
    t02, q02 = target[..., :3], target[..., 3:]
    # q_error = q_target * q_current_inv
    quat_error = math_utils.quat_mul(q02, math_utils.quat_inv(q01))
    # Compute position error
    pos_error = t02 - t01
    return torch.cat([pos_error, quat_error], dim=-1)


@torch.jit.script
def is_pose_converged(error: Tensor, tol: tuple[float, float]) -> BoolTensor:
    """Evaluate whether a pose error has converged according to given tolerances.

    Args:
        error: Pose error between source and target poses. Shape is (..., 7).
        tol: Tolerances for position and rotation errors in meters and radians respectively.

    Return:
        Boolean tensor containing the convergence status of each pose. Shape is (...).
    """
    pos_error = torch.linalg.vector_norm(error[..., :3], dim=-1)
    rot_error = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(error[..., 3:]), dim=-1)
    return torch.logical_and(pos_error < tol[0], rot_error < tol[1])
