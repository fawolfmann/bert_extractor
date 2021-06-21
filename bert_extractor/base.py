"""Extractor base class"""
from abc import ABC
import logging

import pandas as pd  # type: ignore

# from typing import Dict, Optional

logger = logging.getLogger(__name__)

COMPRESSION = "gzip"


class BaseBERTExtractor(ABC):
    def __init__(self):
        """Base class to extract bert classification data from any datasource."""
        self.cache_filepath: str = ""

    def extract(self, url: str) -> pd.DataFrame:
        """Extract classification data, public method to use on every sub class.
        The pipelines is:
            - _extract_raw (here we read it or set the cache)
            - _preprocess
            - _validate_df
        
        Parameters
        ----------
        url : str
            url to extract data from.
        
        Returns
        -------
        pd.DataFrame
            Extracted and preprocessed data to consume BERT model.
        """

    def _extract_raw(self, url: str) -> pd.DataFrame:
        """Extract raw data from a url.
        
        Parameters
        ----------
        url : str
            url to extract data from.
        """

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for Bert Classification problem.
        
        Parameters
        ----------
        df : pd.DataFrame
            raw df extracted.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed df of shape ...
        """

    def _validate_df(self, df: pd.DataFrame):
        """Validate data that will be returned."""

    def _set_cache(self, df: pd.DataFrame):
        """Store extracted dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            dataframe to cache.
        """
        logger.info("Saving to %s", self.cache_filepath)
        df.to_pickle(self.cache_filepath, compression=COMPRESSION)

    def _get_cache(self) -> pd.DataFrame:
        """Read cached data.

        Returns
        -------
        pd.DataFrame
            cached dataframe"""
        logger.info("Reading from %s", self.cache_filepath)
        return pd.read_pickle(self.cache_filepath, compression=COMPRESSION)

    def _set_cache_filepath(self):
        """Set the filepath for cache."""
