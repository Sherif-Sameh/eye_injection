from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from geometry_msgs.msg import TransformStamped
from gymnasium.spaces import Dict
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    subtract_frame_transforms,
)
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_broadcaster import TransformBroadcaster

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCamera
    from isaaclab.utils.noise import NoiseCfg
    from torch import Tensor


@torch.jit.script
def get_robot_relative_transforms(body_poses: Tensor) -> Tensor:
    """Convert the body poses from absolute to relative transforms.

    Converts the body poses that are expressed with respect to the world frame to a kinematic
    chain of relative transforms between each body/link and its parent body/link.

    Args:
        body_poses: Body/link poses relative to the world frame. Shape is (num_bodies, 7).

    Returns:
        Body/link poses relative to their parent body/link. Shape is (num_bodies, 7). For the
            root body/link which has no parent, its transform is kept unchanged.
    """
    num_bodies = body_poses.shape[0]
    rel_pos = torch.clone(body_poses[:, :3])
    rel_rot = torch.clone(body_poses[:, 3:])

    for i in range(1, num_bodies):
        rel_pos[i], rel_rot[i] = subtract_frame_transforms(
            body_poses[i - 1, :3], body_poses[i - 1, 3:], body_poses[i, :3], body_poses[i, 3:]
        )
    body_poses_rel = torch.cat([rel_pos, rel_rot], dim=-1)
    return body_poses_rel


class IsaacLabTFBroadcaster(Node):
    """ROS 2 TF2 transform broadcaster for IsaacLab environments.

    Args:
        env: ManagerBasedRLEnv to interface with ROS 2 using the TF broadcaster node.
        noise: Noise configuration for noise to apply to the true camera extrinsics. Noise for the
            rotation is applied and sampled using the rotation vector representation then the
            resulting orientation is converted back to a quaternion.
    """

    def __init__(self, env: ManagerBasedRLEnv, noise: NoiseCfg | None = None):
        assert env.num_envs == 1
        assert isinstance(env.observation_space, Dict)
        super().__init__("isaac_tf_broadcaster")
        self.noise = noise

        # Initialize transform broadcasters
        self._tf_static_broadcaster = StaticTransformBroadcaster(self)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Publish static transforms once at startup
        self._has_camera = any(
            [len(space.shape) == 4 for space in env.observation_space.spaces.values()]
        )
        if self._has_camera:
            self._make_static_camera_transform(env)

        # Check for existence of tag pose command
        self._has_tag_cmd = "tag_pose" in env.command_manager.active_terms

        # Store robot body names + "world"
        asset: Articulation = env.scene["robot"]
        self._body_names = ["world"] + asset.body_names

    def _make_static_camera_transform(self, env: ManagerBasedRLEnv) -> None:
        """Initialize and send static transform from camera to parent link frame.

        Sends two transforms. A ground truth transform with the child_frame_id of
        "camera_color_optical_frame_gt" and a potentially noise transform with the child_frame_id
        of "camera_color_optical_frame".

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the TF broadcaster node.
        """
        # Retrieve the camera sensor and offset configuration from environment
        asset: TiledCamera = env.scene["camera"]
        offset = asset.cfg.offset

        # Get camera position and orientation in ROS convention (forward axis +Z)
        pos_gt, rot_gt = offset.pos, offset.rot
        if offset.convention != "ros":
            rot_gt = tuple(
                convert_camera_frame_orientation_convention(
                    torch.tensor(rot_gt), origin=offset.convention, target="ros"
                ).tolist()
            )

        # Apply noise to camera pose
        pos_n, rot_n = self._apply_noise_to_camera_transform(pos_gt, rot_gt)
        pos, rot = (pos_n, pos_gt), (rot_n, rot_gt)

        # Setup transformation objects (noisy and GT)
        transforms = []
        for i, (p, r) in enumerate(zip(pos, rot)):
            t = TransformStamped()
            t_sim = env.common_step_counter * env.step_dt
            t.header.stamp = Time(seconds=t_sim).to_msg()
            t.header.frame_id = asset.cfg.prim_path.split("/")[-2]
            t.child_frame_id = "camera_color_optical_frame"
            t.child_frame_id += "_gt" if i == 1 else ""

            t.transform.translation.x = float(p[0])
            t.transform.translation.y = float(p[1])
            t.transform.translation.z = float(p[2])

            t.transform.rotation.w = float(r[0])
            t.transform.rotation.x = float(r[1])
            t.transform.rotation.y = float(r[2])
            t.transform.rotation.z = float(r[3])
            transforms.append(t)

        # Send static transforms
        self._tf_static_broadcaster.sendTransform(transforms)

    def _apply_noise_to_camera_transform(self, pos: tuple, quat: tuple) -> tuple[tuple, tuple]:
        """Apply randomly sampled noise to the given camera transform.

        The configuration of the noise is derived from the instance's `noise` attribute.

        Args:
            pos: Position of the camera relative to its parent frame.
            quat: Quaternion (w, x, y, z) of the camera relative to its parent frame.

        Returns:
            Tuple containing the new position and quaternion after noise application.
        """
        if self.noise is None:
            return pos, quat

        rotvec = R.from_quat(quat, scalar_first=True).as_rotvec()
        pose = torch.from_numpy(np.concatenate([pos, rotvec]))
        pose = self.noise.func(pose, self.noise)
        quat = R.from_rotvec(pose[3:].numpy()).as_quat(scalar_first=True)
        return tuple(pose[:3].tolist()), tuple(quat.tolist())

    def make_static_transforms(self, env: ManagerBasedRLEnv) -> None:
        """Initialize and send static transforms to TF tree."""
        if self._has_camera:
            self._make_static_camera_transform(env)

    def make_robot_transforms(self, env: ManagerBasedRLEnv) -> None:
        """Create and send robot relative transforms between links to TF tree.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the TF broadcaster node.
        """
        # Retrieve the robot and absolute poses from environment
        asset: Articulation = env.scene["robot"]
        body_poses = asset.data.body_link_pose_w[0, :, :]  # Shape (num_bodies, 7)

        # Convert absolute poses to relative
        body_poses_rel = get_robot_relative_transforms(body_poses).cpu()

        # Create transforms for each body
        transforms = []
        t_sim = env.common_step_counter * env.step_dt
        stamp = Time(seconds=t_sim).to_msg()
        for i, pose_rel in enumerate(body_poses_rel):
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self._body_names[i]
            t.child_frame_id = self._body_names[i + 1]

            t.transform.translation.x = float(pose_rel[0])
            t.transform.translation.y = float(pose_rel[1])
            t.transform.translation.z = float(pose_rel[2])

            t.transform.rotation.w = float(pose_rel[3])
            t.transform.rotation.x = float(pose_rel[4])
            t.transform.rotation.y = float(pose_rel[5])
            t.transform.rotation.z = float(pose_rel[6])
            transforms.append(t)

        # Send transforms
        self._tf_broadcaster.sendTransform(transforms)

    def make_tag_transforms(self, env: ManagerBasedRLEnv, cmd: Tensor) -> None:
        """Create and send AprilTag transforms to TF tree if available in command.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the TF broadcaster node.
            cmd: Tensor containing the latest commands from the environment. Shape is (1, cmd_dim).
        """
        if not self._has_tag_cmd:
            return
        cmd = cmd[0].cpu()
        n_tags = cmd.shape[0] // 8

        # Create transforms for each body
        transforms = []
        t_sim = env.common_step_counter * env.step_dt
        stamp = Time(seconds=t_sim).to_msg()
        for i in range(n_tags):
            tag_id = int(cmd[i * 8])
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = f"tag36h11:{tag_id}"
            t.child_frame_id = f"camera_color_optical_frame:{tag_id}"

            t.transform.translation.x = float(cmd[i * 8 + 1])
            t.transform.translation.y = float(cmd[i * 8 + 2])
            t.transform.translation.z = float(cmd[i * 8 + 3])

            t.transform.rotation.w = float(cmd[i * 8 + 4])
            t.transform.rotation.x = float(cmd[i * 8 + 5])
            t.transform.rotation.y = float(cmd[i * 8 + 6])
            t.transform.rotation.z = float(cmd[i * 8 + 7])
            transforms.append(t)

        # Send transforms
        self._tf_broadcaster.sendTransform(transforms)
