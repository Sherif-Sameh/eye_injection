from dataclasses import fields

from isaaclab.sim import PhysxCfg
from isaaclab.utils import configclass


@configclass
class PhysxReducedCfg(PhysxCfg):
    """PhysX configuration for reduced GPU memory usage."""

    partition_reduction: int = 1
    memory_reduction: int = 1
    has_soft_bodies: bool = True
    has_particles: bool = True

    def __post_init__(self):
        assert self.partition_reduction > 0, "Reduction factor must be positive."
        if self.partition_reduction != 1:
            assert self.partition_reduction % 2 == 0, (
                "Reduction factor must be a multiple of 2."
            )
            self.gpu_max_num_partitions = max(
                self.gpu_max_num_partitions / self.partition_reduction, 2
            )

        assert self.memory_reduction > 0, "Reduction factor must be positive."
        if self.memory_reduction != 1:
            assert self.memory_reduction % 2 == 0, (
                "Reduction factor must be a multiple of 2."
            )
            for field in fields(self):
                name = field.name
                if name.startswith("gpu_") and name != "gpu_max_num_partitions":
                    value = getattr(self, name)
                    setattr(self, name, int(value / self.memory_reduction))

        if not self.has_soft_bodies:
            self.gpu_max_soft_body_contacts = 0
        if not self.has_particles:
            self.gpu_max_particle_contacts = 0
