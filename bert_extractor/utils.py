"""Utils"""
from functools import wraps
from hashlib import sha256
import logging
from pathlib import Path
import pickle
from typing import Any, Union

from bert_extractor.extractors.base import TokenizedTensor

logger = logging.getLogger(__name__)


def cache_extract_raw():
    """Cache function results.

    Parameters
    ----------
    cache_path : str
        path to save/load cache.
    name: str
        name of the file to cache.
    cache_read: bool
        True if try to read cache.
    """

    def use_cache_decorator(function):
        """Function result caching wrapper."""

        @wraps(function)
        def wrapper(*args):
            cache_path = args[0].cache_path
            cache_read = args[0].read_cache
            hashed_name = sha256((args[1]).encode()).hexdigest()
            filepath = Path(cache_path) / f"{hashed_name}.pkl"

            if cache_read and filepath.exists():
                result = from_pickle(filepath)
                logger.info("Using cached model: %s.", filepath)
            else:
                result = function(*args)
                Path.mkdir(Path(cache_path), exist_ok=True, parents=True)
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


def store_tensor(tensor: TokenizedTensor, output_path: str, name: str):
    """[summary]

    Parameters
    ----------
    tensor : TokenizedTensor
        [description]
    output_path : str
        [description]
    """
    Path.mkdir(Path(output_path), exist_ok=True, parents=True)

    output_filepath = Path(output_path) / f"{name}_bert_extraction_tensor.pkl"
    to_pickle(output_filepath, tensor)
