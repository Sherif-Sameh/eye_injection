from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .binary_command import BinaryCommand
from .pose_command import PoseCommand
from .tag_pose_command import TagPoseCommand


@configclass
class BinaryCommandCfg(CommandTermCfg):
    """Configuration for the binary command generator."""

    class_type: type = BinaryCommand

    def __post_init__(self):
        """Post initialization."""
        # resampling time range is irrelevant since no resampling takes place
        self.resampling_time_range = (1.0, 1.0)


@configclass
class PoseCommandCfg(CommandTermCfg):
    """Configuration for the pose command generator."""

    class_type: type = PoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    target_prim_names: tuple[str, str] = MISSING
    """Names of the two target primitives for generating pose commands."""

    ref_prim_name: str = MISSING
    """Name of the reference primitive for generating pose commands."""

    binary_command_name: str = MISSING
    """Name of the binary command generator for determining the target asset."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class MotionCfg:
        """Configuration for the desired motion around the target pose."""

        pose_tol: tuple[float, float] = MISSING
        """Tolerance for position and orientation error (in m, rad)."""

        target_offset: float = MISSING
        """Target offset along the target's negative Z-axis (in m)."""

        approach_offset: float = MISSING
        """Approach offset along the target's negative Z-axis (in m)."""

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

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    def __post_init__(self):
        """Post initialization."""
        # resampling time range is irrelevant since no resampling takes place
        self.resampling_time_range = (1.0, 1.0)


@configclass
class TagPoseCommandCfg(CommandTermCfg):
    """Configuration for the tag pose command generator."""

    class_type: type = TagPoseCommand

    camera_asset_name: str = MISSING
    """Name of the source primitive for generating pose commands."""

    tag_prim_names: list[str] = MISSING
    """Names of the target tag primitives for generating pose commands."""

    tag_ids: list[int] = MISSING
    """Corresponding IDs to the target tag primitives of tag_prim_names."""

    pose_ref_prim_name: str = MISSING
    """Name of the reference primitive for which original pose commands are generated."""

    pose_command_name: str = MISSING
    """Name of the pose command generator for determining the original target pose."""

    def __post_init__(self):
        """Post initialization."""
        # resampling time range is irrelevant since no resampling takes place
        self.resampling_time_range = (1.0, 1.0)
