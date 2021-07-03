"""Reviews Data Extractor tests"""

import os
from pathlib import Path
import shutil
from unittest.mock import patch

from pandas.testing import assert_frame_equal

from bert_extractor.extractors import NERExtractor
from tests.extractors.sample_data import (
    extractor_configs,
    ner_extractor_configs,
    ner_sample_df,
    ner_sample_preprocessed,
    ner_txt_sample,
)


def _tests_authentication(
    ner_extractor_configs,
):  # pylint: disable=redefined-outer-name
    """Tests that the environmental variables are set if passed."""
    with patch("bert_extractor.extractors.ner.KaggleApi.authenticate"):
        ner_extractor = NERExtractor(**ner_extractor_configs)

        assert not os.environ.get("KAGGLE_USERNAME")
        assert not os.environ.get("KAGGLE_KEY")
        ner_extractor.authenticate()
        assert (
            os.environ.get("KAGGLE_USERNAME") == ner_extractor_configs["auth_username"]
        )
        assert os.environ.get("KAGGLE_KEY") == ner_extractor_configs["auth_key"]


def _test_raw_extraction_read_concat(
    ner_extractor_configs, ner_txt_sample, ner_sample_df
):  # pylint: disable=redefined-outer-name
    """For given file test that extraction read them and return wanted df."""
    with patch("bert_extractor.extractors.ner.KaggleApi"):

        url = "test_raw_extraction_read_concat"
        download_file_path = Path(f"/tmp/{url}")
        download_file_path.mkdir()
        splits = ["train", "valid", "test"]
        for split in splits:
            file_path = download_file_path / f"{split}.txt"
            file_path.write_text(ner_txt_sample)

        ner_extractor = NERExtractor(**ner_extractor_configs)
        ner_extractor.authenticate()
        df = ner_extractor.extract_raw(url)
        shutil.rmtree(download_file_path, ignore_errors=True)

        assert_frame_equal(df, ner_sample_df)


def _test_raw_extraction_tmp_dir(
    ner_extractor_configs,
):  # pylint: disable=redefined-outer-name
    """Test that a dir not exist after and before the extraction call"""
    with patch("bert_extractor.extractors.ner.KaggleApi"), patch(
        "bert_extractor.extractors.ner.pd"
    ):
        url = "test_raw_extraction_tmp_dir"
        download_file_path = f"/tmp/{url}"
        download_file_path = Path(f"/tmp/{url}")

        assert not os.path.isdir(download_file_path)
        download_file_path.mkdir()
        ner_extractor = NERExtractor(**ner_extractor_configs)
        ner_extractor.authenticate()
        _ = ner_extractor.extract_raw(url)
        assert not os.path.isdir(download_file_path)


def _test_preprocess(
    ner_extractor_configs, ner_sample_df, ner_sample_preprocessed
):  # pylint: disable=redefined-outer-name
    """For a given df test that return preprocessed df"""
    ner_extractor = NERExtractor(**ner_extractor_configs)
    preprocessed_df = ner_extractor.preprocess(ner_sample_df)
    breakpoint()
    assert set(preprocessed_df.columns) == set(
        [ner_extractor_configs["sentence_col"], ner_extractor_configs["labels_col"]]
    )
    assert_frame_equal(preprocessed_df, ner_sample_preprocessed)
