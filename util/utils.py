"""
Some utility functions for the SE training script, which includes:
    is_list: check if an input is a list or not
    load_checkpoint: load a pytorch checkpoint model from a path
    prepare_empty_dir: if the input dir-path doesn't exist, make a new directory.

"""
import random
import importlib
from typing import Union

import json5
import torch
import torch.nn as nn
import numpy as np

from typing import Sequence


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(config_path):
    """
    Load a JSON configuration file.

    Args:
        config_path (pathlib.Path): Path to the configuration file.

    Returns:
        dict: Configuration data.
    """
    with open(config_path) as f:
        config = json5.load(f)
    return config


def replace_denormals(tensor: torch.Tensor, threshold=1e-9) -> torch.Tensor:
    """
    Replaces denormalized values in a tensor with a threshold.

    Args:
        tensor (torch.Tensor): Input tensor.
        threshold (float): Value to replace denormal values with.

    Returns:
        torch.Tensor: Tensor with denormal values replaced.
    """
    return torch.where(tensor.abs() < threshold, threshold, tensor)


def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def initialize_config(module_cfg, pass_args=True):
    """
    According to config items, load specific module dynamically with params.
    eg，config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std
