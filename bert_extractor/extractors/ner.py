"""NER Data Extractor"""
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding

from bert_extractor.constants import NER_LABLES_MAP, SPECIAL_TOKEN_LABEL
from bert_extractor.extractors.base import BaseBERTExtractor
from bert_extractor.utils import cache_extract_raw

logger = logging.getLogger(__name__)


class NERExtractor(BaseBERTExtractor):
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
        """Name Entity Recognition Extractor.
        Extract and preprocess the data for a Token Classification problem,
        now for CoNLL 2003 dataset.

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
        super().__init__(
            pretrained_model_name_or_path,
            sentence_col,
            labels_col,
            auth_username,
            auth_key,
            split_test_size=split_test_size,
            cache_path=cache_path,
            read_cache=read_cache,
        )
        self.api: KaggleApi = None
        self.token_classification = True

    def authenticate(self):
        """Authenticate to Kaggle API.

        Note: there is no way to pass the credentials as parameters to the KaggleApi object.
        """
        if not os.environ.get("KAGGLE_USERNAME"):
            os.environ["KAGGLE_USERNAME"] = self.auth_username
        if not os.environ.get("KAGGLE_KEY"):
            os.environ["KAGGLE_KEY"] = self.auth_key
        self.api = KaggleApi()
        self.api.authenticate()

    @cache_extract_raw()
    def extract_raw(self, url: str) -> Dict:
        """Download the CoNLL 2003 files from Kaggle, into a temporary directory.
        Read them and delete the directory with it content.

        Note:

        Parameters
        ----------
        url : str
            Kaggle dataset name.

        Returns
        -------
        Dict
            sentences_col : List [sentences]
            labels_col : List [sentences]
        """
        logger.info("Going to get data from %s", url)
        download_file = f"/tmp/{url}"
        self.api.dataset_download_files(
            url, path=download_file, unzip=True,
        )
        dataset_types = ["train", "valid", "test"]
        words_all = []
        labels_all = []
        for d_types in dataset_types:
            file_path = f"{download_file}/{d_types}.txt"
            words, labels = self._read_conll_file(file_path=file_path)
            words_all.extend(words)
            labels_all.extend(labels)

        shutil.rmtree(download_file)
        extracted = {self.sentence_col: words_all, self.labels_col: labels_all}
        logger.info("Extraction successfull")
        return extracted

    def _read_conll_file(self, file_path: Union[os.PathLike, str]) -> Tuple[List, List]:
        """Read given file path, supouse to be a CoNLL 2003 file.

        Parameters
        ----------
        file_path : os.PathLike
            path for where is the file.

        Returns
        -------
        Tuple[List, List]
            - words : read words form file.
            - labels : read labels form file.

        Raises
        ------
        ValueError
            if file_path doesn't contain a file.
        """
        words = []
        labels = []

        if not os.path.isfile(file_path):
            error = f"File {file_path} don't exists."
            logger.error(error)
            raise ValueError(error)

        with open(file_path) as file:
            for line in file:
                line = line.rstrip()
                items = line.split(" ")
                words.append(items[0])
                labels.append(items[-1])

        return words, labels

    def preprocess(self, extracted_raw: Dict) -> Tuple[List, List]:
        """Create the columns with the sentences and its labels.
        Concatenate all the sentences and the labels into one line.
        Use separator as a character that don't appers on the files.
        Map labels to integers set in constants.
        Remove -DOCSTART- words.

        Parameters
        ----------
        extracted_raw : Dict
            self.sentence_col : extracted raw words list.
            self.labels_col: extracted raw labels list.

        Returns
        -------
        Tuple[List, List]
            - sentences: list of list of sentences.
            - labels: list of list of mapped labels.

        Raises
        ------
        ValueError
            if the len of the inputs differ.
        """
        words_raw = extracted_raw[self.sentence_col]
        labels_raw = extracted_raw[self.labels_col]

        if len(words_raw) != len(labels_raw):
            error = "Different size of words and labels"
            logger.error(error)
            raise ValueError(error)

        sentences = []
        sentence = []
        label_list = []
        labels = []
        for word, label in zip(words_raw, labels_raw):
            if word:
                if word != "-DOCSTART-":
                    word = word.strip()
                    sentence.append(word)
                    label_list.append(NER_LABLES_MAP.get(label))
            else:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label_list)
                sentence = []
                label_list = []

        logger.info("Preproccessed dataframe")

        return sentences, labels

    def process_labels(
        self, labels: List[List], tokenized_sentences: BatchEncoding
    ) -> np.array:
        """Align and pad labels.
        Pad all labels to the same length that tokens, adding -100 for no tokens.
        Add -100 for `[CLS]` and `[SEP]` tokens.

        Note: BERT can break a word into several so that is needed words_ids.


        Parameters
        ----------
        labels : List[List]
            preprocessed labels.
        tokenized_sentences: BatchEncoding
            Tokenized sentences to use words_ids.

        Returns
        -------
        List
            labels to train a model.
        """
        new_labels = []

        for index, label in enumerate(labels):
            previous_idx = None
            new_label = []
            word_idx = tokenized_sentences.word_ids(batch_index=index)
            for idx in word_idx:
                if idx is None:
                    new_label.append(SPECIAL_TOKEN_LABEL)
                else:
                    if previous_idx == idx:
                        new_label.append(label[idx])
                    else:
                        new_label.append(label[idx])
                previous_idx = idx

            new_labels.append(new_label)

        return np.array(new_labels)
