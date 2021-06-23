"""Extractor base class"""
from abc import ABC
import logging
from pathlib import Path
import pickle
from typing import Any, Union

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

COMPRESSION = "gzip"


class BaseBERTExtractor(ABC):
    def __init__(self):
        """Base class to extract bert classification data from any datasource."""
        self.cache_filepath: Path = None

    def extract_preprocess(self, url: str) -> pd.DataFrame:
        """Extract and preprocess data, for BERT tasks.
        The pipelines is:
            - extract_raw (here we read it from or set the cache)
            - preprocess
            - validate
        
        Parameters
        ----------
        url : str
            url to extract data from.
        
        Returns
        -------
        pd.DataFrame
            Extracted and preprocessed data to consume BERT model.
        """
        raw_df = self.extract_raw(url)
        df = self.preprocess(raw_df)
        self.validate_df(df)

        return df

    def extract_raw(self, url: str) -> pd.DataFrame:
        """Extract raw data from a url.
        If data is cached return cache if not it will download it.
        
        Parameters
        ----------
        url : str
            url to extract data from.
        
        Returns
        -------
        pd.DataFrame
            Extracted data converted to DataFrame.
        """
        return pd.DataFrame()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return pd.DataFrame()

    def validate_df(self, df: pd.DataFrame):
        """Validate that the data satisfy defined criteria.
        
        Parameters
        ----------
        df : pd.DataFrame
            raw df extracted.

        Raises
        ------
        ValueError
            if the data doesn't satisfy some criteria
            
        """

    def _set_cache(self, obj: Any):
        """Pickle raw data.

        Parameters
        ----------
        obj : Any
            data object to store in cache.
        """
        logger.info("Saving to %s", self.cache_filepath)
        with open(self.cache_filepath, "wb") as handle:
            pickle.dump(
                obj, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _get_cache(self) -> Any:
        """Read cached data.

        Returns
        -------
        obj : Any
            cached data
        """
        logger.info("Reading from %s", self.cache_filepath)
        with open(self.cache_filepath, "rb") as handle:
            result = pickle.load(handle)
        return result

    def _set_cache_filepath(self, filename: str, dirpath: Union[Path, str]):
        """Set the filepath for cache.
        Create the directory if not exists.
        TODO add side effect docstring.
        This in a production environment i would dump it to S3 or other storage services.

        Parameters
        ----------
        filepath: str
            name of the file to store.
        dirpath: str or Path
            name or path to the directory to store cache files.
        """
        Path.mkdir(Path(dirpath), exist_ok=True, parents=True)
        self.cache_filepath = Path(dirpath) / filename
