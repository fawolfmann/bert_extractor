"""Extractor base class"""
from abc import ABC
import logging
import os
from typing import NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class TokenizedTensor(NamedTuple):
    """ Tuple of preprocessed tensors."""

    train_inputs: np.array
    validation_inputs: np.array
    train_labels: np.array
    validation_labels: np.array
    train_mask: np.array
    validation_masks: np.array


logger = logging.getLogger(__name__)

COMPRESSION = "gzip"


class BaseBERTExtractor(ABC):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        sentence_col: str,
        labels_col: str,
        auth_username: Optional[str],
        auth_key: Optional[str],
        split_test_size: float = 0.1,
        cache_path: Union[str, os.PathLike] = "/tmp/bert_extractor",
        read_cache: bool = False,
    ):
        """Base class to extract bert classification data from any datasource.

        Parameters
        ----------
        pretrained_model_name_or_path : Union[str, os.PathLike]
            pretained BERT name to tokenize the the input.
        sentence_col : str
            name of the column of from where it will be the text.
        labels_col : str
            name of the column of from where it will be the label.
        auth_username : Optional, str
            username to configure authentication.
        auth_key: Optional, str
            private key to configure authentication.
        split_test_size : float
            amount of dataset to use for test, between [0,1].
        cache_path : Union[str, os.PathLike]
            path to store cached raw data.
        read_cache : bool
            True to read from cache_path
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.sentence_col = sentence_col
        self.labels_col = labels_col
        self.test_size = split_test_size
        self.auth_username = auth_username
        self.auth_key = auth_key
        self.cache_path = cache_path
        self.read_cache = read_cache

    def authenticate(self):
        """Authenticate to a services if needed"""

    def extract_preprocess(self, url: str) -> TokenizedTensor:
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
        TokenizedTensor
            Extracted and preprocessed data to consume BERT model.
        """
        self.authenticate()
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

    def bert_tokenizer(self, df: pd.DataFrame) -> TokenizedTensor:
        """Map the given text to their IDs, prepend the `[CLS]` token to the start,
        append the `[SEP]` token to the end, pad or truncate the sentence to the max text length,
        and create attention masks for [PAD] tokens.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the processed data to tokenize.

        Returns
        -------
            TokenizedTensor tuple of numpy array.

        Note:
            - The parameters bert pretained model named, columns of sentence and label 
            are set in the configs.
            - I use numpy return tensor so that this projects
            isn't dependant on TensorFlow or PyTorch. 

        """
        input_ids = []
        attention_masks = []
        logger.info("Pretrained model name: %s", self.pretrained_model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, do_lower_case=True
        )

        sentences = df[self.sentence_col].values
        labels = df[self.labels_col].values

        tokenized_samples = tokenizer.encode(sentences)
        max_length = min(max(map(len, tokenized_samples)), 512)
        logger.info("Max sentences length %s", max_length)

        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="np",
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
        logger.info("Tokenized %s sentences", len(sentences))

        return TokenizedTensor(
            *train_test_split(
                input_ids,
                labels,
                attention_masks,
                random_state=2020,
                test_size=self.test_size,
            )
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
            if df does not containts labels_col and sentence_col.
            
        """
        if self.labels_col not in df.columns:
            error = f"DataFrame does not contains labels column {self.labels_col} check the configuration or the preprocess method."
            logger.error(error)
            raise ValueError(error)
        if self.sentence_col not in df.columns:
            error = f"DataFrame does not contains sentence column {self.sentence_col} check the configuration or the preprocess method."
            logger.error(error)
            raise ValueError(error)
