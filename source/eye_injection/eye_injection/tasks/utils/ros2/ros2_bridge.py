from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import carb
import numpy as np
import omni
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
import torch
from example_interfaces.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from gymnasium.spaces import Dict
from isaacsim.ros2.bridge import read_camera_info
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectory

from eye_injection.tasks.utils.ros2 import actions

# Warnings disabled from module to prevent repeated harmless warnings
# related to the following issue: https://github.com/isaac-sim/IsaacSim/issues/403
# most likely because simulation stepping is handled through IsaacLab's API instead of IsaacSim
carb.settings.get_settings().set("/log/channels/isaacsim.core.simulation_manager.plugin", "error")

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import TiledCamera
    from isaaclab.utils.noise import NoiseCfg
    from sensor_msgs.msg import CameraInfo
    from torch import Tensor

    from eye_injection.tasks.manager_based.eye_injection.mdp import TrajSmCommand


class IsaacLabRos2Bridge(Node):
    """ROS 2 bridge node for IsaacLab environments.

    The node assumes that the environment's observation space has the following structure:
        Commands: The commands to be published as a Float32MultiArray message*.
        Proprioceptive observations: Joint positions and velocities to be published as a JointState message.
        Extroceptive observations (optional): Camera observations to be published as an Image message.

    *: This the default case. For the `VsTrajCommand` command if it exists, they'll be published
    as a PoseStamped for more clarity. Refer to `_publish_commands_vs()` for msg details.

    Meanwhile, actions are expected to be published through the JointTrajectory message interface.

    Args:
        env: ManagerBasedRLEnv that satifies the above requirements to interface with ROS 2 using
            the bridge node.
        noise: Noise configuration for noise to apply to the true camera intrinsics. Noise is only
            sampled and applied to the three parameters `fx` (=`fy`), `cx` and `cy` of the
            calibration matrix.
    """

    def __init__(self, env: ManagerBasedRLEnv, noise: NoiseCfg | None = None):
        assert env.num_envs == 1
        assert isinstance(env.observation_space, Dict)
        super().__init__("isaac_bridge")
        self.noise = noise

        # Publishers
        self._setup_command_publsher(env)
        self._pub_obs_js = self.create_publisher(JointState, "/isaaclab/joint_states", 0)
        self._pub_rst = self.create_publisher(Empty, "/isaaclab/reset", 1)

        self.has_traj_cmd = "target_traj" in env.command_manager.active_terms
        if self.has_traj_cmd:
            self._pub_perr = self.create_publisher(PoseStamped, "/isaaclab/pose_error", 10)

        has_camera = "camera" in env.scene.keys()
        if has_camera:
            data_types: list[str] = env.scene["camera"].cfg.data_types
            has_depth_type = "depth" in data_types or "distance_to_camera" in data_types
            # Setup camera publishers
            self._setup_observations_image_publisher(env)
            self._setup_camera_info_publisher(env)
            if has_depth_type:
                self._setup_observations_depth_publisher(env)

        # Subscribers
        self._action_fn = self._get_action_fn(env)
        self._sub_acts = self.create_subscription(
            JointTrajectory,
            "/isaaclab/joint_trajectory_controller/joint_trajectory",
            self.action_callback,
            0,
        )

        # Store robot joint attributes
        asset: Articulation = env.scene["robot"]
        self._joint_names = asset.data.joint_names
        self._num_joints = len(self._joint_names)

        # Variables for storing actions, buffer shape = (num_steps, 1, act_dim)
        self._device = env.device
        self._action_ctr = 1
        self._action_buffer = torch.zeros(1, *env.action_space.shape, device=env.device)

    def publish_commands(self, cmd: Tensor) -> None:
        """Publish commands to ROS 2 topic of according to command publisher setup.

        Args:
            cmd: Tensor containing the latest commands to publish. Shape is (1, cmd_dim).
        """
        assert cmd.dtype == torch.float32
        assert cmd.ndim == 2
        cmd = cmd[0].cpu()
        if self._prev_cmd is None or not torch.allclose(cmd, self._prev_cmd):
            self._pub_cmd_fn(cmd)
            self._prev_cmd = cmd.clone()

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

    def publish_pose_error(self, env: ManagerBasedRLEnv) -> None:
        """Publish the computed pose tracking error from the pose command generator.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the bridge node.
        """
        if not self.has_traj_cmd:
            return
        # extract logged pose error metric from environment
        command_term: TrajSmCommand = env.command_manager.get_term("target_traj")
        pose_error = command_term.metrics["pose_error"][0].cpu()

        # create and publish PoseStamped message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(pose_error[0])
        msg.pose.position.y = float(pose_error[1])
        msg.pose.position.z = float(pose_error[2])
        msg.pose.orientation.w = float(pose_error[3])
        msg.pose.orientation.x = float(pose_error[4])
        msg.pose.orientation.y = float(pose_error[5])
        msg.pose.orientation.z = float(pose_error[6])
        self._pub_perr.publish(msg)

    def reset(self, env: ManagerBasedRLEnv) -> None:
        """Publish reset signal and reset camera info publisher if available."""
        msg = Empty()
        self._pub_rst.publish(msg)
        self._prev_cmd = None
        if hasattr(self, "_writer_ci"):
            self._setup_camera_info_publisher(env)

    def action_callback(self, msg: JointTrajectory) -> None:
        """Callback function for action subscriber.

        Joint action sequences are stored in the action buffer and the action counter is reset.

        Args:
            msg: JointTrajectory message sent by controller.
        """
        self._action_ctr = 0
        self._action_buffer = (
            self._action_fn(msg).to(dtype=torch.float32, device=self._device).unsqueeze(1)
        )  # (num_steps, 1, act_dim)

    def get_action(self) -> Tensor:
        """Get the current action to apply to environment from the action buffer.

        Note: if all actions in the buffer have already been previously applied, the last action is
        repeated until new actions are recieved and the buffer is updated.

        Returns:
            Tensor containing the action to apply to environment. Shape is (1, act_dim).
        """
        idx = min(self._action_ctr, self._action_buffer.shape[0] - 1)
        action = self._action_buffer[idx]
        self._action_ctr += 1
        return action

    @staticmethod
    def _get_action_fn(env: ManagerBasedRLEnv) -> Callable[[JointTrajectory], Tensor]:
        """Get the appropriate action function according to the environment's action space.

        Action functions are used to extract the appropriate fields from a JointTrajectory msg
        into a Tensor. All valid combinations of joint position, velocity and effort control are
        supported. Check for function for valid combinations.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the bridge node.

        Returns:
            Callable function that given a JointTrajectory msg will return a Tensor of joint
                actions extracted appropriately from the msg.
        """
        # Determine active joint control terms
        action_terms = env.action_manager.active_terms
        has_position = "joint_position" in action_terms
        has_velocity = "joint_velocity" in action_terms
        has_effort = "joint_effort" in action_terms
        assert has_position or has_velocity or has_effort, (
            f"Env has no recognizable active action terms. Got {action_terms}."
        )

        # Joint position control (velocity and/or effort feed-forward allowed)
        if has_position:
            if not has_velocity and not has_effort:
                return actions.position_action
            if has_velocity and not has_effort:
                return actions.position_action_velocity_ff
            if not has_velocity and has_effort:
                return actions.position_action_effort_ff
            return actions.position_action_velocity_effort_ff

        # Joint velocity control (effort feed-forward allowed)
        if has_velocity:
            if not has_effort:
                return actions.velocity_action
            return actions.velocity_action_effort_ff

        # Joint effort control (no feed-forward allowed)
        return actions.effort_action

    def _setup_command_publsher(self, env: ManagerBasedRLEnv) -> None:
        """Setup command publisher and function according to the environment's config."""
        has_vs_cmd = "vs_traj" in env.command_manager.active_terms
        if not has_vs_cmd:
            cls = Float32MultiArray
            fn = self._publish_commands_default
        else:
            cls = PoseStamped
            fn = self._publish_commands_vs
        self._prev_cmd = None
        self._pub_cmd = self.create_publisher(
            cls,
            "/isaaclab/command",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self._pub_cmd_fn = fn

    def _publish_commands_default(self, cmd: Tensor) -> None:
        """Publish commands to ROS 2 topic of type Float32MultiArray.

        Args:
            cmd: Tensor containing the latest commands to publish. Shape is (cmd_dim,).
        """
        msg = Float32MultiArray()
        msg.data = cmd.tolist()
        self._pub_cmd.publish(msg)

    def _publish_commands_vs(self, cmd: Tensor) -> None:
        """Publish pose command to ROS 2 topic of type PoseStamped.

        Args:
            cmd: Tensor containing the latest state commands to publish. Shape is (13,).
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = float(cmd[0])
        msg.pose.position.y = float(cmd[1])
        msg.pose.position.z = float(cmd[2])
        msg.pose.orientation.w = float(cmd[3])
        msg.pose.orientation.x = float(cmd[4])
        msg.pose.orientation.y = float(cmd[5])
        msg.pose.orientation.z = float(cmd[6])
        self._pub_cmd.publish(msg)

    def _setup_observations_image_publisher(self, env: ManagerBasedRLEnv) -> None:
        """Setup publisher for rgb image observations through IsaacSim.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the bridge node.
        """
        camera: TiledCamera = env.scene["camera"]
        render_product = camera.render_product_paths[0]

        # Link the camera's render product and publish the data to the specified topic name
        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.Rgb.name
        )
        writer = rep.writers.get(rv + "ROS2PublishImage")
        writer.initialize(
            frameId="camera_color_optical_frame",
            nodeNamespace="",
            queueSize=0,
            topicName="/isaaclab/camera/image_raw",
        )
        writer.attach([render_product])

    def _setup_observations_depth_publisher(self, env: ManagerBasedRLEnv) -> None:
        """Setup publisher for depth image observations through IsaacSim.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the bridge node.
        """
        camera: TiledCamera = env.scene["camera"]
        render_product = camera.render_product_paths[0]

        # Link the camera's render product and publish the data to the specified topic name
        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
            sd.SensorType.DistanceToImagePlane.name
        )
        writer = rep.writers.get(rv + "ROS2PublishImage")
        writer.initialize(
            frameId="camera_depth_aligned_optical_frame",
            nodeNamespace="",
            queueSize=0,
            topicName="/isaaclab/camera/depth_raw",
        )
        writer.attach([render_product])

    def _setup_camera_info_publisher(self, env: ManagerBasedRLEnv) -> None:
        """Setup publisher for camera's information through IsaacSim.

        The camera's intrinsics are set after the application of the randomly sampled noise to the
        camera's true intrinsic parameters.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the bridge node.
        """
        camera: TiledCamera = env.scene["camera"]
        render_product = camera.render_product_paths[0]

        # The following code will link the camera's render product and publish the data to the specified topic name.
        if not hasattr(self, "_writer_ci"):
            self._writer_ci = rep.writers.get("ROS2PublishCameraInfo")
        else:
            self._writer_ci.detach()
        camera_info, _ = read_camera_info(render_product_path=render_product)
        self._writer_ci.initialize(
            frameId="camera_color_optical_frame",
            nodeNamespace="",
            queueSize=0,
            topicName="/isaaclab/camera/camera_info",
            width=camera_info.width,
            height=camera_info.height,
            projectionType=camera_info.distortion_model,
            k=self._apply_noise_to_camera_intrinsics(camera_info),
            r=camera_info.r.reshape([1, 9]),
            p=camera_info.p.reshape([1, 12]),
            physicalDistortionModel=camera_info.distortion_model,
            physicalDistortionCoefficients=camera_info.d,
        )
        self._writer_ci.attach([render_product])

    def _apply_noise_to_camera_intrinsics(self, camera_info: CameraInfo) -> None:
        """Apply randomly sampled noise to the stored camera intrinsic parameters.

        Returns:
            Camera calibration matrix after noise application. Shape is (1, 9).
        """
        if self.noise is None:
            return camera_info.k.reshape([1, 9])

        params = torch.tensor([camera_info.k[0], camera_info.k[2], camera_info.k[5]])
        params = self.noise.func(params, self.noise)
        return np.array([params[0], 0, params[1], 0, params[0], params[2], 0, 0, 1]).reshape([1, 9])
