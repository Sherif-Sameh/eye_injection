from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from example_interfaces.msg import Float32MultiArray
from gymnasium.spaces import Dict
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory

from isaaclab_assets import UR10e_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
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
            Float32MultiArray, "/isaaclab/cmd/target_eye", 10
        )
        self._pub_obs_js = self.create_publisher(
            JointState, "/isaaclab/obs/joint_states", 10
        )
        if any(
            [len(space.shape) == 4 for space in env.observation_space.spaces.values()]
        ):
            self._pub_obs_img = self.create_publisher(Image, "/isaaclab/obs/image", 10)

        # Action subscriber
        self._sub_acts = self.create_subscription(
            JointTrajectory,
            "isaaclab/act/joint_torques",
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
