"""Reviews Data Extractor tests"""

import os
from pathlib import Path
import shutil
from unittest.mock import mock_open, patch

from pandas.testing import assert_frame_equal

from bert_extractor.extractors.ner import NERExtractor
from tests.extractors.sample_data import (
    extractor_configs,
    ner_extractor_configs,
    ner_sample_preprocessed,
    ner_sample_raw,
    ner_txt_sample,
)


def tests_authentication(ner_extractor_configs,):
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


def test_raw_extraction_read_concat(
    ner_extractor_configs, ner_txt_sample, ner_sample_raw
):
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
        extracted_raw = ner_extractor.extract_raw(url)
        shutil.rmtree(download_file_path, ignore_errors=True)

        assert extracted_raw == ner_sample_raw


def test_raw_extraction_tmp_dir(ner_extractor_configs,):
    """Test that a dir not exist after and before the extraction call"""
    with patch("bert_extractor.extractors.ner.KaggleApi"), patch(
        "bert_extractor.extractors.ner.NERExtractor._read_conll_file"
    ) as file_reader:
        try:
            file_reader.return_value = ([], [])
            url = "test_raw_extraction_tmp_dir"
            download_file_path = f"/tmp/{url}"
            download_file_path = Path(f"/tmp/{url}")

            assert not os.path.isdir(download_file_path)
            download_file_path.mkdir()
            ner_extractor = NERExtractor(**ner_extractor_configs)
            ner_extractor.authenticate()
            _ = ner_extractor.extract_raw(url)
            assert not os.path.isdir(download_file_path)

        finally:
            if os.path.isdir(download_file_path):
                shutil.rmtree(download_file_path)


def test_preprocess(ner_extractor_configs, ner_sample_raw, ner_sample_preprocessed):
    """For a given df test that return preprocessed df"""
    ner_extractor = NERExtractor(**ner_extractor_configs)
    preprocessed_data = ner_extractor.preprocess(ner_sample_raw)

    assert len(preprocessed_data) == 2
    assert preprocessed_data == ner_sample_preprocessed
