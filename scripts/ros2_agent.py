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
parser.add_argument("-s", "--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("-n", "--n_runs", type=int, default=0, help="Number of episodes to simulate.")
parser.add_argument("-c", "--config", type=str, default="base.toml", help="Config file to load.")
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

from pathlib import Path
from typing import Any

import eye_injection.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import rclpy
import torch
from eye_injection.tasks.utils.common import (
    apply_overrides,
    load_toml,
    seed_everything,
    to_noise_cfg,
)
from eye_injection.tasks.utils.ros2 import IsaacLabRos2Bridge, IsaacLabTFBroadcaster
from isaaclab_tasks.utils import parse_env_cfg


def load_config() -> dict[str, Any]:
    """Load environment configuration with optional overrides applied to defaults."""
    # load environment config file
    config = load_toml(Path(__file__).parent / f"config/{args_cli.config}")
    # load default environment configuration
    env_cfg = parse_env_cfg(config["task_name"])
    # override seed if given in args
    if args_cli.seed is not None:
        config["env"]["seed"] = args_cli.seed
    # convert any noise cfgs descriptions to instances of NoiseCfg
    to_noise_cfg(config)
    # apply overrides from config file
    config["env"] = apply_overrides(env_cfg, config["env"])
    return config


def main():
    """ROS2 actions agent with Isaac Lab environment."""
    # load configuration and set seed for reproducable results
    config = load_config()
    seed_everything(config["env"].seed)

    # create and reset environment
    env = gym.make(config["task_name"], cfg=config["env"])
    obs, info = env.reset()

    # initialize ROS and create ROS 2 bridge and tf broadcaster nodes
    rclpy.init()
    bridge_node = IsaacLabRos2Bridge(env.unwrapped, **config["ros2_bridge"])
    tf_broadcaster_node = IsaacLabTFBroadcaster(env.unwrapped, **config["tf_broadcaster"])

    # simulate environment
    autoreset = False
    n_runs_left = args_cli.n_runs if args_cli.n_runs > 0 else float("inf")
    while simulation_app.is_running() and n_runs_left > 0:
        # run everything in inference mode
        with torch.inference_mode():
            # publish commands, observations, pose errors and transforms to ROS 2
            bridge_node.publish_commands(obs["policy_cmd"])
            bridge_node.publish_observations_jointstate(obs["policy_prop"])
            bridge_node.publish_pose_error(env.unwrapped)
            if autoreset:
                n_runs_left -= 1
                bridge_node.reset(env.unwrapped)
                tf_broadcaster_node.make_static_transforms(env.unwrapped)
            tf_broadcaster_node.make_robot_transforms(env.unwrapped)
            tf_broadcaster_node.make_tag_transforms(env.unwrapped, obs["policy_cmd"])

            # update ROS to check for published actions
            rclpy.spin_once(bridge_node, timeout_sec=0)

            # apply action get observations from environment
            action = bridge_node.get_action()
            obs, _, terminated, truncated, _ = env.step(action)
            autoreset = torch.logical_or(terminated, truncated)

    # close the simulator
    env.close()

    # close the ROS 2 bridge
    bridge_node.destroy_node()
    tf_broadcaster_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
