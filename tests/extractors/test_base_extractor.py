"""BaseBERTExtractor tests"""

import pytest


@pytest.fixture
def dummy_dict():
    return {"a": 1}


def test_dummy(dummy_dict):
    assert dummy_dict == {"a": 1}
