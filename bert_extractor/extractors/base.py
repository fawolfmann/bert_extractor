"""Extractor base class"""
from abc import ABC
import logging
import os
from typing import Any, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TokenizedTensor(NamedTuple):
    """ Tuple of preprocessed tensors."""

    train_inputs: BatchEncoding
    validation_inputs: BatchEncoding
    train_labels: np.array
    validation_labels: np.array


class BaseBERTExtractor(ABC):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        sentence_col: str,
        labels_col: str,
        auth_username: Optional[str] = None,
        auth_key: Optional[str] = None,
        split_test_size: float = 0.1,
        cache_path: Union[str, os.PathLike] = "/tmp/bert_extractor",
        read_cache: bool = False,
    ):
        """Base class to extract BERT classification data from any datasource.

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
        self.token_classification = False

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
        extracted = self.extract_raw(url)
        sentences, labels = self.preprocess(extracted)

        return self.bert_tokenizer(sentences, labels)

    def extract_raw(self, url: str) -> Any:
        """Extract raw data from a url.
        If data is cached return cache if not it will download it.

        Parameters
        ----------
        url : str
            url to extract data from.

        Returns
        -------
        Any
            extracted raw data.
        """
        return {}

    def preprocess(self, extracted_raw: Any) -> Tuple[List, List]:
        """Preprocess data for BERT Classification problem.

        Parameters
        ----------
        extracted_raw: Any
            extracted raw data on any format.

        Returns
        -------
        Tuple[List, List]
            - sentences: preprocessed sentences.
            - labels: preprocessed labels.
        """
        return (extracted_raw[self.sentence_col], extracted_raw[self.labels_col])

    def bert_tokenizer(self, sentences: List, labels: List) -> TokenizedTensor:
        """Map the given text to their IDs, prepend the `[CLS]` token to the start,
        append the `[SEP]` token to the end, pad or truncate the sentence to the max text length,
        and create attention masks for [PAD] tokens.

        Parameters
        ----------
        sentences : List
            sentences to tokenize.
        labels: List
            labels to processes if needed.

        Returns
        -------
            TokenizedTensor tuple of numpy array.

        Note:
            - The parameters BERT pretained model named is set in the configs.
            - I use numpy return tensor so that this project
            isn't dependant on TensorFlow or PyTorch.

        """
        logger.info("Pretrained model name: %s", self.pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, do_lower_case=True, use_fast=True,
        )
        if self.token_classification:
            max_length = self._round_nearst_pow(
                min(
                    max(len(tokenizer.encode(" ".join(sent))) for sent in sentences),
                    512,
                )
            )
        else:
            max_length = self._round_nearst_pow(
                min(
                    max(len(tokenizer.encode(sent)) for sent in sentences),
                    tokenizer.model_max_length,
                )
            )
        logger.info("Max sentences length %s", max_length)
        train_sentences, val_sentences, train_labels, val_labels = train_test_split(
            sentences, labels, random_state=2020, test_size=self.test_size,
        )
        train_tokenized, train_labels = self._tokenize_split(
            train_sentences, train_labels, max_length, tokenizer
        )
        val_tokenized, val_labels = self._tokenize_split(
            val_sentences, val_labels, max_length, tokenizer
        )

        return TokenizedTensor(
            train_inputs=train_tokenized,
            validation_inputs=val_tokenized,
            train_labels=train_labels,
            validation_labels=val_labels,
        )

    def _tokenize_split(
        self,
        sentences: List[str],
        labels: List,
        max_length: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Tuple[BatchEncoding, List]:
        """Helper function to tokenize and align and pad sentences and labels.

        Parameters
        ----------
        sentences : List[str]
            list of sentences to tokenize.
        labels : List
            list of labels to process.
        max_length : int
            max length of the encoded sentences.
        tokenizer : PreTrainedTokenizerBase
            tokenizer created to process the sentences.

        Returns
        -------
        Tuple[BatchEncoding, List]
            - tokenized: tokenized sentences to use with BERT model.
            - labels : np.array processed labels

        """
        tokenized = tokenizer(
            sentences,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            is_split_into_words=self.token_classification,
            return_tensors="np",
        )

        labels = self.process_labels(labels, tokenized)

        return tokenized, labels

    def _round_nearst_pow(self, number: int) -> int:
        """Round max length to a higher power of 8 to power up NVIDIA GPUs.

        Parameters
        ----------
        number : int
            number to round

        Returns
        -------
        int
            rounded number.
        """
        return (number + 7) & (-8)

    def process_labels(
        self, labels: List, tokenized_sentences: BatchEncoding
    ) -> np.array:
        """Process labels if needed.

        Parameters
        ----------
        labels : List
            labels to process
        words_ids : List
            id of each token corresponding to a label.

        Returns
        -------
        np.array
            processed labels in as numpy.array.
        """
        return np.array(labels)
