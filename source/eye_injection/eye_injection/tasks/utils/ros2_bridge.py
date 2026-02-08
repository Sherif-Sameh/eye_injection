from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from example_interfaces.msg import Float32MultiArray
from gymnasium.spaces import Dict
from isaaclab.sim import PinholeCameraCfg
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectory

from isaaclab_assets import UR10e_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCamera
    from torch import Tensor


class IsaacLabRos2Bridge(Node):
    """ROS 2 bridge node for IsaacLab environments.

    The node assumes that the environment's observation space has the following structure:
        Commands: The commands to be published as a Float32MultiArray message.
        Proprioceptive observations: Joint positions and velocities to be published as a JointState message.
        Extroceptive observations (optional): Camera images to be published as an Image message.

    Meanwhile, actions are expected to be published as joint torques through the JointTrajectory
    message interface.

    Args:
        env: ManagerBasedRLEnv that satifies the above requirements to interface with ROS 2 using
            the bridge node.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        assert env.num_envs == 1
        assert isinstance(env.observation_space, Dict)
        super().__init__("isaac_bridge")

        # Command and observations publishers
        self._pub_cmd = self.create_publisher(
            Float32MultiArray, "/isaaclab/command", 10
        )
        self._pub_obs_js = self.create_publisher(
            JointState, "/isaaclab/joint_states", 10
        )
        if any(
            [len(space.shape) == 4 for space in env.observation_space.spaces.values()]
        ):  # has image observations
            # Camera publishers
            self._pub_obs_img = self.create_publisher(
                Image, "/isaaclab/camera/image_raw", 10
            )
            self._pub_cam_info = self.create_publisher(
                CameraInfo, "/isaaclab/camera/camera_info", 10
            )

            # Initialize camera information and setup timer for publisher
            self._init_camera_info(env)
            self._timer_cam_info = self.create_timer(1.0, self.publish_camera_info)

        # Action subscriber
        self._sub_acts = self.create_subscription(
            JointTrajectory,
            "isaaclab/joint_trajectory_controller/joint_trajectory",
            self.action_callback,
            10,
        )

        # Store joint attributes
        self._joint_names = list(UR10e_CFG.init_state.joint_pos.keys())
        self._num_joints = len(self._joint_names)

        # Variables for storing actions (num_steps, 1, act_dim)
        self._device = env.device
        self._action_ctr = 1
        self._action_zero = torch.zeros(*env.action_space.shape, device=env.device)
        self._action_buffer = torch.zeros(1, *env.action_space.shape, device=env.device)

    def _init_camera_info(self, env: ManagerBasedRLEnv) -> None:
        """Initialize the camera's information to ROS 2 CameraInfo message.

        This function is meant to be used only once during initialization. A CameraInfo message is
        prepared and stored for use by the publisher's timer callback.

        Args:
            env: ManagerBasedRLEnv that satifies the above requirements to interface with ROS 2 using
            the bridge node.
        """
        # Retrieve the camera sensor and configuration from environment
        asset: TiledCamera = env.scene["camera"]
        pinhole_cfg: PinholeCameraCfg = asset.cfg.spawn
        assert isinstance(pinhole_cfg, PinholeCameraCfg), (
            f"Expected pinhole camera config, got {type(pinhole_cfg)}."
        )

        # Setup CameraInfo message
        self._msg_cam_info = CameraInfo()

        # Basic properties
        self._msg_cam_info.height = asset.cfg.height
        self._msg_cam_info.width = asset.cfg.width
        self._msg_cam_info.distortion_model = "plumb_bob"

        # Intrinsic camera matrix (K)
        # [fx, 0,  cx,
        #  0,  fy, cy,
        #  0,  0,  1]
        pixel_size = pinhole_cfg.horizontal_aperture / float(asset.cfg.width)
        fx = pinhole_cfg.focal_length / pixel_size  # fx = fy
        self._msg_cam_info.k = [fx, 0.0, 0.0, 0.0, fx, 0.0, 0.0, 0.0, 1.0]

        # Distortion coefficients (no distortion)
        self._msg_cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Rectification Matrix (3x3 identity for monocular cameras)
        self._msg_cam_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection Matrix ([K | 0] for monocular cameras)
        # [fx', 0,   cx', Tx,
        #  0,   fy', cy', Ty,
        #  0,   0,   1,   0]
        self._msg_cam_info.p = self._msg_cam_info.k.tolist() + [0.0, 0.0, 0.0]

    def publish_commands(self, cmd: Tensor) -> None:
        """Publish commands to ROS 2 topic of type Float32MultiArray.

        Args:
            cmd: Tensor containing the latest commands to publish. Shape is (1, cmd_dim).
        """
        assert cmd.dtype == torch.float32
        assert cmd.ndim == 2
        cmd = cmd[0].cpu()

        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self._pub_cmd.publish(msg)

    def publish_observations_jointstate(self, obs: Tensor) -> None:
        """Publish joint state observations to ROS 2 topic of type JointState.

        Args:
            obs: Tensor containing the latest joint state observations to publish.
                Shape is (1, 2 * num_joints).
        """
        assert obs.dtype == torch.float32
        assert obs.ndim == 2 and obs.shape[1] == self._num_joints * 2
        obs = obs[0].cpu()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._joint_names
        msg.position = obs[: self._num_joints].tolist()
        msg.velocity = obs[self._num_joints :].tolist()
        self._pub_obs_js.publish(msg)

    def publish_observations_image(self, obs: Tensor) -> None:
        """Publish image observations to ROS 2 topic of type Image.

        Args:
            obs: Tensor containing the latest image observations to publish.
                Shape is (1, H, W, 3).
        """
        assert obs.dtype == torch.uint8
        assert obs.ndim == 4 and obs.shape[3] == 3
        obs = obs[0].cpu().contiguous()

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = obs.shape[0]
        msg.width = obs.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3  # bytes per row
        msg.data = obs.numpy().tobytes()
        self._pub_obs_img.publish(msg)

    def publish_camera_info(self) -> None:
        """Publish the camera's information to ROS 2 topic of type CameraInfo.

        Note: **Must** be called after the camera's information has been initialized.
        """
        # Set the header's timestamp and publish ready-made message
        self._msg_cam_info.header.stamp = self.get_clock().now().to_msg()
        self._pub_cam_info.publish(self._msg_cam_info)

    def action_callback(self, msg: JointTrajectory) -> None:
        """Callback function for action subscriber.

        Joint torque sequences are stored in the action buffer and the action counter is reset.

        Args:
            msg: JointTrajectory message sent by controller.
        """
        self._action_ctr = 0
        self._action_buffer = torch.tensor(
            [point.effort for point in msg.points],
            dtype=torch.float32,
            device=self.env_device,
        ).unsqueeze(1)

    def get_action(self) -> Tensor:
        """Get the current action to apply to environment from the action buffer.

        Note: if the action buffer is empty, the zero action is applied instead.

        Returns:
            Tensor containing the action to apply to environment. Shape is (1, act_dim).
        """
        if self._action_ctr < self._action_buffer.shape[0]:
            action = self._action_buffer[self.action_ctr]
            self._action_ctr += 1
        else:
            action = self._action_zero
        return action
