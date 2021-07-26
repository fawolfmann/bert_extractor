"""Reviews Data Extractor"""

import csv
import logging
from typing import List, Tuple

import numpy as np
from transformers.tokenization_utils_base import BatchEncoding

from bert_extractor.constants import COVID_TWEETS_LABLES_MAP
from bert_extractor.extractors.base import BaseBERTExtractor
from bert_extractor.utils import cache_extract_raw

logger = logging.getLogger(__name__)


class CSVExtractor(BaseBERTExtractor):
    """Extractor for Amazon Reviews"""

    @cache_extract_raw()
    def extract_raw(self, local_path: str) -> List:
        """Read the text and labels from a csv file.

        Parameters
        ----------
        local_path : str
            local_path for .csv file to extract.

        Returns
        -------
        List
            list with all the data extracted.
        """
        with open(local_path, mode="r", errors="ignore") as csv_file:
            loaded_dict = list(csv.DictReader(csv_file, dialect="unix"))

        logger.info("Extraction successfull")
        return loaded_dict

    def preprocess(self, extracted_data: List) -> Tuple[List, List]:
        """Create two lists with the sentences and labels.

        Parameters
        ----------
        extracted_data : List
            extracted raw data.

        Returns
        -------
        Tuple[List, List]
            - list of raw words.
            - list of raw labels.
        """
        sentences = []
        labels = []
        for raw in extracted_data:
            sentences.append(raw.get("OriginalTweet", ""))
            labels.append(COVID_TWEETS_LABLES_MAP.get(raw["Sentiment"]))

        logger.info("Preproccessed dataframe")

        return sentences, labels

    def process_labels(
        self, labels: List, tokenized_sentences: BatchEncoding
    ) -> np.array:
        """Process labels as in this problem the labels are numbers from 1 to 5.
        Here just subtract 1 and ensure int type.

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

        return np.array(labels).astype(int)
