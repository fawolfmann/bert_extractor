"""Utils"""
import json
import logging
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def use_cache(config: Dict, url: str):
    """Cache function results.

    Parameters
    ----------
    config : Dict
        configurations

    to use: @use_cache(config=configs)
    """

    def use_cache_decorator(function):
        """Function result caching wrapper."""

        @wraps(function)
        def wrapper(*args, **kwargs):

            filepath = Path(config["cache_filepath"]) / f"{url}.pkl"

            if config["read_cache"] and filepath.exists():
                result = from_pickle(filepath)
                logger.info("Using cached model: %s.", filepath)
            else:
                result = function(*args, **kwargs)
                if config["active"]:
                    Path.mkdir(
                        Path(config["cache_filepath"]), exist_ok=True, parents=True
                    )
                    to_pickle(filepath, result)
                    logger.info("Cached model to: %s.", filepath)
            return result

        return wrapper

    return use_cache_decorator


def to_pickle(filepath: Union[str, Path], obj: Any):
    """Pickle object."""
    with open(filepath, "wb") as handle:
        pickle.dump(
            obj, handle, protocol=pickle.HIGHEST_PROTOCOL,
        )


def from_pickle(filepath: Union[str, Path]):
    """Load pickled object."""
    with open(filepath, "rb") as handle:
        result = pickle.load(handle)
    return result


def store_tensor(tensor: Tuple[np.array], output_path: str, name: str):
    """[summary]

    Parameters
    ----------
    tensor : Tuple[np.array]
        [description]
    output_path : str
        [description]
    """
    Path.mkdir(Path(output_path), exist_ok=True, parents=True)

    output_filepath = Path(output_path) / f"{name}_bert_extraction_tensor.pkl"
    to_pickle(output_filepath, tensor)

