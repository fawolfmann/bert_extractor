"""NER Data Extractor"""
import csv
import logging

import pandas as pd

from bert_extractor.extractors.base import BaseBERTExtractor

logger = logging.getLogger(__name__)


class NERExtractor(BaseBERTExtractor):
    def extract_raw(self, url: str) -> pd.DataFrame:
        """Download the txt files from Kaggle.

        Note: 

        Parameters
        ----------
        url : str
            url from the json.gz data to download.

        Returns
        -------
        pd.DataFrame
            df with all the data extracted.
        """
        logger.info("Going to get data from %s", url)

        df = pd.read_csv(
            "../bert_extras/kaggle/test.txt",
            sep=" ",
            skip_blank_lines=False,
            header=None,
            quoting=csv.QUOTE_NONE,
        )
        df.columns = ["text", "POS", "POS2", "tag"]

        logger.info("Extraction successfull")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the columns with the sentences and labels.
        Delete -DOCSTART- rows
        Parameters
        ----------
        df : pd.DataFrame
            extracted raw data.

        Returns
        -------
        pd.DataFrame
            processed df.
        """
        df = df[df.text != "-DOCSTART-"]
        separetor = "|||"
        df[self.sentence_col] = df.text.str.cat(sep=" ", na_rep=separetor).split(
            separetor
        )
        df[self.labels_col] = df.tag.str.cat(sep=" ", na_rep=separetor).split(separetor)
        df = df[[self.labels_col, self.sentence_col]]
        df.dropna(inplace=True)
        logger.info("Preproccessed dataframe")

        return df
