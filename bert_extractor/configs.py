"""Utils"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Union

from bert_extractor.constants import (
    KNOWN_CONFIGS_TYPES,
    KNOWN_NER_URLS,
    KNOWN_REVIEWS_URLS,
    NER_CONFIG_TYPE,
    REVIEWS_CONFIG_TYPE,
)

logger = logging.getLogger(__name__)


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
    logger.info("Reading config: %s", config_path)
    with open(config_path, "r") as file:
        config = json.load(file)

    _validate_config(config)

    return config


def _validate_config(config: Dict):
    """Validate configs are set properly.

    Parameters
    ----------
    config : Dict
        Read configurations.
    
    Raise
    -----
        ValueError if ani validation does not fullfil. 
    """
    if config.get("extractor_type") not in KNOWN_CONFIGS_TYPES:
        error_message = f"Unknown extractor type, knows {KNOWN_CONFIGS_TYPES}"
        logger.error(error_message)
        raise ValueError(error_message)

    if config.get("extractor_type") == NER_CONFIG_TYPE:
        if config.get("extractor_url") not in KNOWN_NER_URLS:
            error_message = f"Unknown extractor type, knows {KNOWN_NER_URLS}"
            logger.error(error_message)
            raise ValueError(error_message)

    elif config.get("extractor_type") == REVIEWS_CONFIG_TYPE:
        if config.get("extractor_url") not in KNOWN_REVIEWS_URLS:
            error_message = f"Unknown extractor type, knows {KNOWN_REVIEWS_URLS}"
            logger.error(error_message)
            raise ValueError(error_message)

