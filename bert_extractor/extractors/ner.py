"""NER Data Extractor"""
import csv
import logging
import os
import shutil
from typing import Optional, Union

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

from bert_extractor.constants import NER_LABLES_MAP
from bert_extractor.extractors.base import BaseBERTExtractor

logger = logging.getLogger(__name__)


class NERExtractor(BaseBERTExtractor):
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

    def extract_raw(self, url: str) -> pd.DataFrame:
        """Download the txt files from Kaggle, into a temporary directory.
        Read them and delete the directory with it content.

        Note: 

        Parameters
        ----------
        url : str
            Kaggle dataset name.

        Returns
        -------
        pd.DataFrame
            df with all the data extracted.
        """
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
        shutil.rmtree(download_file)
        full_extracted_df = pd.concat(dfs)
        full_extracted_df.reset_index(drop=True, inplace=True)

        logger.info("Extraction successfull")
        return full_extracted_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the columns with the sentences and labels.
        Concatenate all the sentences and the labels into one line.
        Use separator as a character that don't appers on the files.
        Map labels to integers set in constants.
        Remove '-DOCSTART-' lines.

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
        df_out = pd.DataFrame()
        df_out[self.sentence_col] = df.text.str.cat(sep=" ", na_rep=separetor).split(
            separetor
        )
        df_out[self.labels_col] = df.tag.str.cat(sep=" ", na_rep=separetor).split(
            separetor
        )
        # iterate over every row map and return a list.
        df["tag"] = df_out[self.labels_col].map(NER_LABLES_MAP)
        df_out.dropna(inplace=True)
        logger.info("Preproccessed dataframe")

        return df_out
