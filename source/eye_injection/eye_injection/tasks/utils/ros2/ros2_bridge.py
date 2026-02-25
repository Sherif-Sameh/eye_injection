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
from scipy.spatial.transform import Rotation as R
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

    from eye_injection.tasks.manager_based.eye_injection.mdp import PoseCommand


class IsaacLabRos2Bridge(Node):
    """ROS 2 bridge node for IsaacLab environments.

    The node assumes that the environment's observation space has the following structure:
        Commands: The commands to be published as a Float32MultiArray message.
        Proprioceptive observations: Joint positions and velocities to be published as a JointState message.
        Extroceptive observations (optional): Camera images to be published as an Image message.

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
        self._pub_cmd = self.create_publisher(Float32MultiArray, "/isaaclab/command", 0)
        self._pub_obs_js = self.create_publisher(JointState, "/isaaclab/joint_states", 0)
        self._pub_perr = self.create_publisher(PoseStamped, "/isaaclab/pose_error", 10)
        self._pub_rst = self.create_publisher(Empty, "/isaaclab/reset", 0)

        has_camera = any([len(space.shape) == 4 for space in env.observation_space.spaces.values()])
        if has_camera:
            # Setup camera publishers
            self._setup_observations_image_publisher(env)
            self._setup_camera_info_publisher(env)

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
        self._action_zero = torch.zeros(*env.action_space.shape, device=env.device)
        self._action_buffer = torch.zeros(1, *env.action_space.shape, device=env.device)

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

    def publish_pose_error(self, env: ManagerBasedRLEnv) -> None:
        """Publish the computed pose tracking error from the pose command generator.

        Args:
            env: ManagerBasedRLEnv to interface with ROS 2 using the bridge node.
        """
        # extract logged pose error metric from environment
        command_term: PoseCommand = env.command_manager.get_term("target_pose")
        pos_error = command_term.metrics["position_error"][0].cpu()
        rot_error = command_term.metrics["rotation_error"][0].cpu()
        rot_error = R.from_rotvec(rot_error).as_quat()  # (x, y, z, w)

        # create and publish PoseStamped message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(pos_error[0])
        msg.pose.position.y = float(pos_error[1])
        msg.pose.position.z = float(pos_error[2])
        msg.pose.orientation.x = float(rot_error[0])
        msg.pose.orientation.y = float(rot_error[1])
        msg.pose.orientation.z = float(rot_error[2])
        msg.pose.orientation.w = float(rot_error[3])
        self._pub_perr.publish(msg)

    def reset(self, env: ManagerBasedRLEnv) -> None:
        """Publish reset signal and reset camera info publisher if available."""
        msg = Empty()
        self._pub_rst.publish(msg)
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

        Note: if the action buffer is empty, the zero action is applied instead.

        Returns:
            Tensor containing the action to apply to environment. Shape is (1, act_dim).
        """
        if self._action_ctr < self._action_buffer.shape[0]:
            action = self._action_buffer[self._action_ctr]
            self._action_ctr += 1
        else:
            action = self._action_zero
        return action
