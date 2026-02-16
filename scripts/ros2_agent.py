# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with ROS2-based agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="ROS2 agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# enable ROS 2 extension
from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import rclpy
import torch
from example_interfaces.msg import Float32MultiArray
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory

from isaaclab_assets import UR10e_CFG  # isort:skip
import eye_injection.tasks  # noqa: F401
from eye_injection.tasks.utils import IsaacLabRos2Bridge, IsaacLabTFBroadcaster
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """ROS2 actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    # create and reset environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, info = env.reset()

    # initialize ROS and create ROS 2 bridge and tf broadcaster nodes
    rclpy.init()
    bridge_node = IsaacLabRos2Bridge(env.unwrapped)
    tf_broadcaster_node = IsaacLabTFBroadcaster(env.unwrapped)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # publish commands, observations, pose errors and transforms to ROS 2
            bridge_node.publish_commands(obs["policy_cmd"])
            bridge_node.publish_observations_jointstate(obs["policy_prop"])
            bridge_node.publish_pose_error(env.unwrapped)
            tf_broadcaster_node.make_robot_transforms(env.unwrapped)

            # update ROS to check for published actions
            rclpy.spin_once(bridge_node, timeout_sec=0)

            # apply action get observations from environment
            action = bridge_node.get_action()
            obs, rewards, terminated, truncated, info = env.step(action)

    # close the simulator
    env.close()

    # close the ROS 2 bridge
    bridge_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
