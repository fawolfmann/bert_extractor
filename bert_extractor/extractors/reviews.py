"""Reviews Data Extractor"""

from gzip import decompress
import json

import pandas as pd
import requests

from bert_extractor.extractors.base import BaseBERTExtractor


class ReviewsExtractor(BaseBERTExtractor):
    def extract_raw(self, url: str) -> pd.DataFrame:
        """Download the url for Amazon reviews cast to a DataFrame.

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
        pd.DataFrame
            df with all the data extracted.
        """
        return pd.DataFrame(
            json.loads(
                "["
                + decompress(requests.get(url).content)
                .decode("utf-8")
                .replace("}\n{", "},{")
                + "]"
            )
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """[summary]

        Parameters
        ----------
        df : pd.DataFrame
            [description]

        Returns
        -------
        pd.DataFrame
            [description]
        """
        df["sentence"] = df["summary"] + " : " + df["reviewText"]
        # REVIEW THIS LINE, see if we can remove it and it dont break the dropna()
        df = df[["overall", "sentence"]]
        df.dropna(inplace=True)
        df["overall"] = df["overall"].astype(int)

        return df
