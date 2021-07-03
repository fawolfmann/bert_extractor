"""NER Data Extractor"""
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np

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
        """[summary]

        Parameters
        ----------
        pretrained_model_name_or_path : Union[str, os.PathLike]
            [description]
        sentence_col : str
            [description]
        labels_col : str
            [description]
        auth_username : Optional[str]
            [description]
        auth_key : Optional[str]
            [description]
        split_test_size : float
            [description]
        cache_path : Union[str, os.PathLike]
            [description]
        read_cache : bool
            [description]
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
        # CHECK IF THIS IS OK
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
        df_splits = ["train", "valid", "test"]
        words_all = []
        labels_all = []
        for splits in df_splits:
            file_path = f"{download_file}/{splits}.txt"
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
            words_all.extend(words)
            labels_all.extend(labels)

        shutil.rmtree(download_file)
        extracted = {self.sentence_col: words_all, self.labels_col: labels_all}
        logger.info("Extraction successfull")
        return extracted

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

    def process_labels(self, labels: List[List], words_ids: List[List]) -> np.array:
        """Align and pad labels.
        Pad all labels to the same length that tokens, adding -100 for no tokens.
        Add -100 for `[CLS]` and `[SEP]` tokens.


        Parameters
        ----------
        labels : List[List]
            preprocessed labels.
        max_length: int
            maximum length of tokens.

        Returns
        -------
        List
            labels to train a model.
        """
        new_labels = []

        for label, word_idx in zip(labels, words_ids):
            previous_idx = None
            new_label = []
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
