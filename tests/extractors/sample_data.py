"""Bert Data Extractor"""

import pandas as pd
import pytest


@pytest.fixture
def extractor_configs():
    return {
        "pretrained_model_name_or_path": "bert-base-uncased",
        "sentence_col": "text",
        "labels_col": "label",
    }


@pytest.fixture()
def ner_extractor_configs(extractor_configs):
    return {
        "auth_username": "tests_user",
        "auth_key": "tests_key",
        **extractor_configs,
    }


@pytest.fixture
def sample_preprocessed():
    return (
        [
            "Five Stars : As advertised. Reasonably priced",
            "Good for the face : Like the oder and the feel when I put it on my face.  I have tried other brands but the reviews from people I know they prefer the oder of this brand. Not hard on the face when dry.  Does not leave dry skin.",
        ],
        [5.0, 5.0],
    )


@pytest.fixture
def sample_extracted():
    return [
        {
            "overall": 5.0,
            "verified": True,
            "reviewTime": "09 1, 2016",
            "reviewerID": "A3CIUOJXQ5VDQ2",
            "asin": "B0000530HU",
            "style": {"Size:": " 7.0 oz", "Flavor:": " Classic Ice Blue"},
            "reviewerName": "Shelly F",
            "reviewText": "As advertised. Reasonably priced",
            "summary": "Five Stars",
            "unixReviewTime": 1472688000,
        },
        {
            "overall": 5.0,
            "verified": True,
            "reviewTime": "11 14, 2013",
            "reviewerID": "A3H7T87S984REU",
            "asin": "B0000530HU",
            "style": {"Size:": " 7.0 oz", "Flavor:": " Classic Ice Blue"},
            "reviewerName": "houserules18",
            "reviewText": "Like the oder and the feel when I put it on my face.  I have tried other brands but the reviews from people I know they prefer the oder of this brand. Not hard on the face when dry.  Does not leave dry skin.",
            "summary": "Good for the face",
            "unixReviewTime": 1384387200,
        },
    ]


@pytest.fixture
def ner_sample_raw():
    return {
        "text": [
            "-DOCSTART-",
            "",
            "SOCCER",
            "JAPAN",
            "WIN",
            ",",
            "-DOCSTART-",
            "",
            "SOCCER",
            "JAPAN",
            "WIN",
            ",",
            "-DOCSTART-",
            "",
            "SOCCER",
            "JAPAN",
            "WIN",
            ",",
        ],
        "label": [
            "O",
            "",
            "O",
            "B-LOC",
            "O",
            "O",
            "O",
            "",
            "O",
            "B-LOC",
            "O",
            "O",
            "O",
            "",
            "O",
            "B-LOC",
            "O",
            "O",
        ],
    }


@pytest.fixture
def ner_sample_preprocessed():
    return (
        [["SOCCER", "JAPAN", "WIN", ","], ["SOCCER", "JAPAN", "WIN", ","]],
        [[9, 1, 9, 9], [9, 1, 9, 9]],
    )


@pytest.fixture
def ner_txt_sample():
    text = """-DOCSTART- -X- -X- O

SOCCER NN B-NP O
JAPAN NNP B-NP B-LOC
WIN NNP I-NP O
, , O O"""
    return text
