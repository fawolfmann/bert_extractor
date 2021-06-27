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
