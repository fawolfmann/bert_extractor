"""Extractor base class"""
from abc import ABC
import logging
import os
from pathlib import Path
import pickle
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

COMPRESSION = "gzip"


class BaseBERTExtractor(ABC):
    def __init__(
        self,
        cache_filepath: Path,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        sentence_col: str,
        labels_col: str,
    ):
        """Base class to extract bert classification data from any datasource."""
        self.cache_filepath = cache_filepath
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.sentence_col = sentence_col
        self.labels_col = labels_col

    def extract_preprocess(
        self, url: str
    ) -> Tuple[
        np.array, np.array, np.array, np.array, np.array, np.array,
    ]:
        """Extract and preprocess data, for BERT tasks.
        The pipelines is:
            - extract_raw (here we read it from or set the cache)
            - preprocess
            - bert_tokenizer
            - validate
        
        Parameters
        ----------
        url : str
            url to extract data from.
        
        Returns
        -------
        Tuple[np.array]
            Extracted and preprocessed data to consume BERT model.
        """
        raw_df = self.extract_raw(url)
        df = self.preprocess(raw_df)
        self.validate_df(df)

        return self.bert_tokenizer(df)

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

    def bert_tokenizer(
        self, df: pd.DataFrame
    ) -> Tuple[
        np.array, np.array, np.array, np.array, np.array, np.array,
    ]:
        """[summary]

        Parameters
        ----------
        df : pd.DataFrame
            [description]
        sentence_col: str
            [description]
        labels_col: str
            [description]

        Returns
        -------
        Tuple[
            train_inputs : ,
            validation_inputs : ,
            train_labels : ,
            validation_labels : ,
            train_mask: ,
            validation_masks : ,
        ]
        """
        tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, do_lower_case=True
        )

        sentences = df[self.sentence_col].values
        labels = df[self.labels_col].values

        input_ids = []
        attention_masks = []
        # TODO define max_length depending the input.

        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                padding="max_length",
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="np",  # Return numpy tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict["input_ids"])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict["attention_mask"])

        return train_test_split(
            input_ids, labels, attention_masks, random_state=2020, test_size=0.2
        )

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
