from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .binary_command import BinaryCommand


@configclass
class BinaryCommandCfg(CommandTermCfg):
    """Configuration for the binary command generator."""

    class_type: type = BinaryCommand

    def __post_init__(self):
        """Post initialization."""
        # resampling time range is irrelevant since no resampling takes place
        self.resampling_time_range = (1.0, 1.0)
