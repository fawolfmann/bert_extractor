"""NER Data Extractor"""
import csv
import logging
import os
from typing import Union

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

from bert_extractor.extractors.base import BaseBERTExtractor

logger = logging.getLogger(__name__)


class NERExtractor(BaseBERTExtractor):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        sentence_col: str,
        labels_col: str,
        split_test_size: float,
        cache_path: Union[str, os.PathLike],
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
        split_test_size : float
            [description]
        cache_path : Union[str, os.PathLike]
            [description]
        """
        super().__init__(
            pretrained_model_name_or_path,
            sentence_col,
            labels_col,
            split_test_size=split_test_size,
            cache_path=cache_path,
        )
        self.api = None

    def authenticate(self, username: str, key: str):
        """[summary]

        Parameters
        ----------
        username : str
            [description]
        key : str
            [description]
        """
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        self.api = KaggleApi()
        self.api.authenticate()

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
        url = "alaakhaled/conll003-englishversion"
        logger.info("Going to get data from %s", url)
        download_file = f"/tmp/{url}"
        self.api.dataset_download_files(
            url, path=download_file, unzip=True,
        )
        df_splits = ["train", "valid", "test"]
        dfs = []
        for splits in df_splits:
            df = pd.read_csv(
                f"{download_file}/{splits}.txt",
                sep=" ",
                skip_blank_lines=False,
                header=None,
                quoting=csv.QUOTE_NONE,
            )
            df.columns = ["text", "POS", "POS2", "tag"]
            dfs.append(df)

        full_extracted_df = pd.concat(dfs)

        logger.info("Extraction successfull")
        return full_extracted_df

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
        # TODO MAP labels to ints
        df[self.labels_col] = df.tag.str.cat(sep=" ", na_rep=separetor).split(separetor)
        df = df[[self.labels_col, self.sentence_col]]
        df.dropna(inplace=True)
        logger.info("Preproccessed dataframe")

        return df
