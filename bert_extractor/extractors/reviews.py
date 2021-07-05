"""Reviews Data Extractor"""

from gzip import decompress
import json
import logging
from typing import List, Tuple

import numpy as np
import requests
from transformers.tokenization_utils_base import BatchEncoding

from bert_extractor.extractors.base import BaseBERTExtractor
from bert_extractor.utils import cache_extract_raw

logger = logging.getLogger(__name__)


class ReviewsExtractor(BaseBERTExtractor):
    """Extractor for Amazon Reviews"""

    @cache_extract_raw()
    def extract_raw(self, url: str) -> List:
        """Download the url for Amazon reviews cast to a dict.

        Note: the unzipped string containts jsons bad formated, here we cast them to one df.
        example of raw data:
        "{"overall":5.0, "reviewText": " awesome product"}
        {"overall":1.0, "reviewText": "worst product, it was borken"}
        {"overall":5.0, "reviewText": "my dad love it"}
        "

        Parameters
        ----------
        url : str
            url from the json.gz data to download.

        Returns
        -------
        List
            list with all the data extracted.
        """
        logger.info("Going to get data from %s", url)
        loaded_dict = json.loads(
            "["
            + decompress(requests.get(url).content)
            .decode("utf-8")
            .replace("}\n{", "},{")
            + "]"
        )

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
            sentences.append(raw.get("summary", "") + " : " + raw.get("reviewText", ""))
            labels.append(raw["overall"])

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
        return np.array(labels).astype(int) - 1
