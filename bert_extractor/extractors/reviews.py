"""Reviews Data Extractor"""

from gzip import decompress
import json
import logging
from typing import Dict, List, Tuple

import requests

from bert_extractor.extractors.base import BaseBERTExtractor

logger = logging.getLogger(__name__)


class ReviewsExtractor(BaseBERTExtractor):
    def extract_raw(self, url: str) -> Dict:
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
        Dict
            dict with all the data extracted.
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

    def preprocess(self, loaded_dict: Dict) -> Tuple[List, List]:
        """Create two lists with the sentences and labels.

        Parameters
        ----------
        loaded_dict : Dict
            extracted raw data.

        Returns
        -------
        Tuple[List, List]
            - list of raw words.
            - list of raw labels.
        """
        sentences = []
        labels = []
        for raw in loaded_dict:
            sentences.append(raw.get("summary", "") + " : " + raw.get("reviewText", ""))
            labels.append(raw["overall"])

        logger.info("Preproccessed dataframe")

        return sentences, labels
