"""BaseBERTExtractor tests"""

import numpy as np
import pytest

from bert_extractor.extractors.base import BaseBERTExtractor
from tests.extractors.sample_data import extractor_configs, sample_preprocessed


def test_bert_tokenizer_output(extractor_configs, sample_preprocessed):
    """Test the output, size of validation dataset,
    test mask tensor are binnary and the size of the tokenized input,
    the second element of the dataframe is the biggest but is smaller than 512
    so it have to be the same length plus the [CLS] and [SEP] tokens.
    """
    extractor_configs["split_test_size"] = 0.5
    base = BaseBERTExtractor(**extractor_configs)
    tensor = base.bert_tokenizer(*sample_preprocessed)

    assert len(tensor) == 4
    assert (
        len(tensor.validation_inputs["input_ids"])
        == len(sample_preprocessed) * base.test_size
    )
    assert set(np.unique(tensor.validation_inputs["attention_mask"][0])) == set([0, 1])
    assert (
        tensor.validation_inputs["input_ids"][0].shape[0]
        == tensor.train_inputs["input_ids"][0].shape[0]
    )


def test_bert_tokenizer_model_name(extractor_configs, sample_preprocessed):
    """Test model_name, a incorrect name,
    this should never happen, this is validated on configs,
    but better to be prepare"""
    extractor_configs["pretrained_model_name_or_path"] = "wrong-bert-name"
    base = BaseBERTExtractor(**extractor_configs)

    with pytest.raises(OSError):
        _ = base.bert_tokenizer(*sample_preprocessed)
