from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .binary_command import BinaryCommand
from .traj_sm_command import TrajSmCommand
from .vs_traj_command import VsTrajCommand


@configclass
class BinaryCommandCfg(CommandTermCfg):
    """Configuration for the binary command generator."""

    class_type: type = BinaryCommand

    prob_1: float = MISSING

    def __post_init__(self):
        """Post initialization."""
        # resampling time range is irrelevant since its disabled internally by the command
        self.resampling_time_range = (1.0, 1.0)


@configclass
class TrajSmCommandCfg(CommandTermCfg):
    """Configuration for the FSM-based trajectory state command generator."""

    class_type: type = TrajSmCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which trajectory commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which trajectory commands are generated."""

    target_prim_names: tuple[str, str] = MISSING
    """Names of the two target primitives for extracting target poses."""

    ref_prim_name: str = MISSING
    """Name of the reference primitive for extracting target poses."""

    binary_command_name: str = MISSING
    """Name of the binary command generator for determining the target asset."""

    @configclass
    class MotionCfg:
        """Configuration for the desired motion around the target pose."""

        pose_tol: tuple[float, float] = MISSING
        """Tolerance for position and orientation error (in m, rad)."""

        target_offset: float = MISSING
        """Target pose offset along the target's negative Z-axis (in m)."""

        approach_offset: float = MISSING
        """Approach pose offset along the target's negative Z-axis (in m)."""

        approach_vel: float = MISSING
        """Approach velocity along the target's Z-axis (in m/s)."""

        stationary_time: float = MISSING
        """Stationary time at the target pose. (in sec)."""

        retreat_vel: float = MISSING
        """Retreat velocity along the target's negative Z-axis (in m/s)."""

    motion_cfg: MotionCfg = MISSING
    """Motion configuration."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.04, 0.04, 0.04)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)

    def __post_init__(self):
        """Post initialization."""
        # resampling time range is irrelevant since since its disabled internally by the command
        self.resampling_time_range = (1.0, 1.0)


@configclass
class VsTrajCommandCfg(CommandTermCfg):
    """Configuration for the visual servoing trajectory command generator."""

    class_type: type = VsTrajCommand

    ref_prim_names: tuple[str, str] = MISSING
    """Names of the two reference primitives for generating target poses."""

    pose_ref_prim_name: str = MISSING
    """Name of the reference primitive for which original state commands are generated."""

    traj_command_name: str = MISSING
    """Name of the trajectory command generator for determining the original target state."""

    binary_command_name: str = MISSING
    """Name of the binary command generator for determining the reference primitive."""

    def __post_init__(self):
        """Post initialization."""
        # resampling time range is irrelevant since no resampling takes place
        self.resampling_time_range = (1.0, 1.0)
