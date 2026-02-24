import os
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import tomllib
import torch
from isaaclab.utils import noise


def seed_everything(seed: int | None) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: The seed value to use.
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load the toml file from the given path.

    Args:
        path: Path to the toml file to load.

    Returns:
        Output dictionary from loading the toml file.
    """
    path = Path(path) if isinstance(path, str) else path
    assert path.exists(), f"Path {path} does not exist."
    assert path.suffix == ".toml", f"Expected a .toml file, got {path.suffix}"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data


def to_noise_cfg(cfg: dict[str, Any]) -> None:
    """Convert named descriptions of noise to instances of `NoiseCfg` in configuration dict.

    Iterates over a configuration of potentially nested dicts and replaces any named instances of
    IsaacLab `NoiseCfg` classes with instantiated instances of those classes using the given
    parameters.

    Each `NoiseCfg` entry is expected to be a dict with the following entries:
        - noise_class (str): Name of the noise class which is a subclass of `NoiseCfg`.
        - parameters (dict): Optional dictionary of parameters to pass the constructor of the
        given noise class.

    Args:
        cfg: Configuration dictionary to traverse and modify any `NoiseCfg` entries.
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            if "noise_class" in value:
                cls = getattr(noise, value["noise_class"])
                assert issubclass(cls, noise.NoiseCfg)
                params = value.get("parameters", {})
                cfg[key] = cls(**params)
            else:
                to_noise_cfg(cfg[key])


def apply_overrides(cfg: object, overrides: dict[str, Any]) -> object:
    """Apply overrides to a `dataclass` configuration from given dictionary.

    Args:
        cfg: Configuration instance of a dataclass to apply given overrides to.
        overrides: Dictionary of overrides to apply to configuration.

    Returns:
        Modified configuration with applied overrides.
    """
    for field in fields(cfg):
        name = field.name
        if name not in overrides:
            continue

        value = getattr(cfg, name)
        if hasattr(value, "__dict__"):
            if type(value) is type(overrides[name]):
                setattr(cfg, name, overrides[name])
            else:
                setattr(cfg, name, apply_overrides(value, overrides[name]))
        elif type(value) is dict:
            setattr(cfg, name, apply_overrides_dict(value, overrides[name]))
        else:
            if type(value) is tuple and type(overrides[name]) is list:
                overrides[name] = tuple(overrides[name])
            setattr(cfg, name, overrides[name])
    return cfg


def apply_overrides_dict(
    cfg: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Apply overrides to a configuration dictionary from given dictionary.

    **Warning**: Nested dicts in the configuration are ignored and not overriden.
    Dataclasses are set to their given values without unwrapping of their fields.

    Args:
        cfg: Configuration dictionary to apply given overrides to.
        overrides: Dictionary of overrides to apply to configuration.

    Returns:
        Modified configuration with applied overrides.
    """
    for name, value in cfg.items():
        if name not in overrides or type(value) is dict:
            continue

        if type(value) is tuple and type(overrides[name]) is list:
            overrides[name] = tuple(overrides[name])
        cfg[name] = overrides[name]
    return cfg
