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
    return pd.DataFrame(
        {
            "label": {0: 5, 1: 5},
            "text": {
                0: "Five Stars : As advertised. Reasonably priced",
                1: "Good for the face : Like the oder and the feel when I put it on my face.  I have tried other brands but the reviews from people I know they prefer the oder of this brand. Not hard on the face when dry.  Does not leave dry skin.",
            },
        }
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "overall": {0: 5.0, 1: 5.0},
            "verified": {0: True, 1: True},
            "reviewTime": {0: "09 1, 2016", 1: "11 14, 2013"},
            "reviewerID": {0: "A3CIUOJXQ5VDQ2", 1: "A3H7T87S984REU"},
            "asin": {0: "B0000530HU", 1: "B0000530HU"},
            "style": {
                0: {"Size:": " 7.0 oz", "Flavor:": " Classic Ice Blue"},
                1: {"Size:": " 7.0 oz", "Flavor:": " Classic Ice Blue"},
            },
            "reviewerName": {0: "Shelly F", 1: "houserules18"},
            "reviewText": {
                0: "As advertised. Reasonably priced",
                1: "Like the oder and the feel when I put it on my face.  I have tried other brands but the reviews from people I know they prefer the oder of this brand. Not hard on the face when dry.  Does not leave dry skin.",
            },
            "summary": {0: "Five Stars", 1: "Good for the face"},
            "unixReviewTime": {0: 1472688000, 1: 1384387200},
        }
    )


@pytest.fixture
def ner_sample_df():
    return pd.DataFrame(
        {
            "text": {
                0: "-DOCSTART-",
                1: None,
                2: "SOCCER",
                3: "JAPAN",
                4: "WIN",
                5: ",",
                6: "-DOCSTART-",
                7: None,
                8: "SOCCER",
                9: "JAPAN",
                10: "WIN",
                11: ",",
                12: "-DOCSTART-",
                13: None,
                14: "SOCCER",
                15: "JAPAN",
                16: "WIN",
                17: ",",
            },
            "POS": {
                0: "-X-",
                1: None,
                2: "NN",
                3: "NNP",
                4: "NNP",
                5: ",",
                6: "-X-",
                7: None,
                8: "NN",
                9: "NNP",
                10: "NNP",
                11: ",",
                12: "-X-",
                13: None,
                14: "NN",
                15: "NNP",
                16: "NNP",
                17: ",",
            },
            "POS2": {
                0: "-X-",
                1: None,
                2: "B-NP",
                3: "B-NP",
                4: "I-NP",
                5: "O",
                6: "-X-",
                7: None,
                8: "B-NP",
                9: "B-NP",
                10: "I-NP",
                11: "O",
                12: "-X-",
                13: None,
                14: "B-NP",
                15: "B-NP",
                16: "I-NP",
                17: "O",
            },
            "tag": {
                0: "O",
                1: None,
                2: "O",
                3: "B-LOC",
                4: "O",
                5: "O",
                6: "O",
                7: None,
                8: "O",
                9: "B-LOC",
                10: "O",
                11: "O",
                12: "O",
                13: None,
                14: "O",
                15: "B-LOC",
                16: "O",
                17: "O",
            },
        }
    )


@pytest.fixture
def ner_sample_preprocessed():
    return pd.DataFrame()


@pytest.fixture
def ner_txt_sample():
    text = """-DOCSTART- -X- -X- O

SOCCER NN B-NP O
JAPAN NNP B-NP B-LOC
WIN NNP I-NP O
, , O O"""
    return text
