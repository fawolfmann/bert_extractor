"""Utils"""


import os
from typing import Dict, Tuple, Union

import numpy as np


def cache_wrapper():
    pass


def read_config(config_path: Union[str, os.PathLike]) -> Dict:
    """Parse and validate configuration

    Parameters
    ----------
    config_path : Union[str, os.PathLike]
        Path for the config file.

    Returns
    -------
    Dict : dictionary with the configuration.
    """

    return {}


def store_tensor(tensor: Tuple[np.array], output_path: str, output_format: str):
    """[summary]

    Parameters
    ----------
    tensor : Tuple[np.array]
        [description]
    """
    return
