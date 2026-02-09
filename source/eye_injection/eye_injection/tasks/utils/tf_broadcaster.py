from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from geometry_msgs.msg import TransformStamped
from gymnasium.spaces import Dict
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    subtract_frame_transforms,
)
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_broadcaster import TransformBroadcaster

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCamera
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
            body_poses[i - 1, :3],
            body_poses[i - 1, 3:],
            body_poses[i, :3],
            body_poses[i, 3:],
        )
    body_poses_rel = torch.cat([rel_pos, rel_rot], dim=-1)
    return body_poses_rel


class IsaacLabTFBroadcaster(Node):
    """ROS 2 TF2 transform broadcaster for IsaacLab environments.

    Args:
        env: ManagerBasedRLEnv to interface with ROS 2 using the TF broadcaster node.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        assert env.num_envs == 1
        assert isinstance(env.observation_space, Dict)
        super().__init__("isaac_tf_broadcaster")

        # Initialize transform broadcasters
        self._tf_static_broadcaster = StaticTransformBroadcaster(self)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Publish static transforms once at startup
        has_camera = any(
            [len(space.shape) == 4 for space in env.observation_space.spaces.values()]
        )
        if has_camera:
            self._make_static_camera_transform(env)

        # Store robot body names + "world"
        asset: Articulation = env.scene["robot"]
        self._body_names = ["world"] + asset.body_names

    def _make_static_camera_transform(self, env: ManagerBasedRLEnv) -> None:
        """Initialize and send static transform from camera to parent link frame.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the TF broadcaster node.
        """
        # Retrieve the camera sensor and offset configuration from environment
        asset: TiledCamera = env.scene["camera"]
        offset = asset.cfg.offset

        # Get camera position and orientation in ROS convention (forward axis +Z)
        pos = offset.pos
        rot = offset.rot
        if offset.convention != "ros":
            rot = convert_camera_frame_orientation_convention(
                torch.tensor(rot),
                origin=offset.convention,
                target="ros",
            )

        # Setup transformation object
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = asset.cfg.prim_path.split("/")[-2]
        t.child_frame_id = "camera_optical_color_frame"

        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])

        t.transform.rotation.w = float(rot[0])
        t.transform.rotation.x = float(rot[1])
        t.transform.rotation.y = float(rot[2])
        t.transform.rotation.z = float(rot[3])

        # Send static transform
        self._tf_static_broadcaster.sendTransform(t)

    def make_robot_transforms(self, env: ManagerBasedRLEnv):
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
        stamp = self.get_clock().now().to_msg()
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
