"""Reviews Data Extractor tests"""

from unittest.mock import patch

from bert_extractor.extractors.reviews import ReviewsExtractor
from tests.extractors.sample_data import (
    extractor_configs,
    sample_extracted,
    sample_preprocessed,
)


def test_raw_extraction_request(extractor_configs,):
    """Test request object is called with the correct url."""
    with patch("requests.get") as requests:
        url = ""
        requests.return_value.content = b"\x1f\x8b\x08\x00\xe5\xaf\xd7`\x02\xff\xbdQMO\xe3@\x0c\xbd\xef\xaf\xb0rFQ\xd2\xb4$\xf4\xb6P\xbat\x85@\xf4c\xc5u\xda8\xca\xa8\x93q\xe5\x99\xb4\x84\x8a\xff\xbe\x9eF*h\x97\x0b\x17r\xca\xbcg\xfb=?\x1f#\xda#+c\xa21\x8c\xe2\xe4\x02\"y\xeaJc)\x80\xe7\x16\x05a\xdck<,u\x83\x82E\xc9\x15\xa4\x170H\xd2\xcb\xe8\xcc!\xcf&\x81\xfb\x99\xdd\xccV\x8f\xbf\x9f\x9fF\x7f&O\x83\xc0+\xa7m`\xae\x13\xf9FYr\xb7\n\xa8\xf3\x9d\t\xc3\x8e\xd1B\xbf\xe28\x14@\x1e'@\xaf\x81\x9d\x1a\xb5'\xee\xd1\x1b\xa3\x9c\xd3\x1b\x98m\x10\xaeM\x8b\xd1\xdb\x07\xd1\x07\xd5[Z\xd4hL\x07\xd3wCK|\xf1'C\x0eT)\x1by\xed\xb0\x8ca\x8e\xca\x91Uk)\xde\xb1\xde\xc8\x92\xc1L\xdb4\x8a\xbbP>\xd5{\x84\x85W\xec\x02\xd1Z\xfd2\xff\xb8{:\xcc\x07\x97E!\x9b\xbc\xfd8~9\xb84\x95\x01\xa7\xe4\xb2\xcf\x92\xbb\xcb\x97E\xbe\xb8*\x86\xf3\xdb\xd5\xb7%WS\xeb\x90[\x83.-\xfeO\xef^o\x11|\x8d@%2([\x9e\x1e\x15\xa2\x81C\x8d\x16f\xb0k=h\x0fd\xa1\xe9\xa0R\x1b\x8cA\xd0ZI\x8c\x9e%\x0b \xe9`X\xb34;XKu\x98\xd0\xab8\xa8\x98\x1a\xd8!\xed\x0cJ\xd7\xd6\xd2!\xd0\xe16XI\xd7Y\x99*\xf9\xd7\xae\x1f\x13\xc3\x03y\x91\xe02\xc8\x9e\x0c\x89no\xa8\xe4N\x0cL\x08\x1dX)2\x18\x8c\x08\x08n\xabm\xfc\xcf\xb1\x7f\x11\x95P\x11\x9fg|~\xf3\xac\x18fE>\x90\x9b\xff\x05\x07\x1bt[,\x03\x00\x00"
        reviews_extractor = ReviewsExtractor(**extractor_configs)
        _ = reviews_extractor.extract_raw(url)

        requests.assert_called_once_with(url)


def test_raw_extraction_decompress(extractor_configs, sample_extracted):
    """Test the extracted data is decompressed and return as expected."""
    with patch("requests.get") as requests:
        requests.return_value.content = b"\x1f\x8b\x08\x00\xe5\xaf\xd7`\x02\xff\xbdQMO\xe3@\x0c\xbd\xef\xaf\xb0rFQ\xd2\xb4$\xf4\xb6P\xbat\x85@\xf4c\xc5u\xda8\xca\xa8\x93q\xe5\x99\xb4\x84\x8a\xff\xbe\x9eF*h\x97\x0b\x17r\xca\xbcg\xfb=?\x1f#\xda#+c\xa21\x8c\xe2\xe4\x02\"y\xeaJc)\x80\xe7\x16\x05a\xdck<,u\x83\x82E\xc9\x15\xa4\x170H\xd2\xcb\xe8\xcc!\xcf&\x81\xfb\x99\xdd\xccV\x8f\xbf\x9f\x9fF\x7f&O\x83\xc0+\xa7m`\xae\x13\xf9FYr\xb7\n\xa8\xf3\x9d\t\xc3\x8e\xd1B\xbf\xe28\x14@\x1e'@\xaf\x81\x9d\x1a\xb5'\xee\xd1\x1b\xa3\x9c\xd3\x1b\x98m\x10\xaeM\x8b\xd1\xdb\x07\xd1\x07\xd5[Z\xd4hL\x07\xd3wCK|\xf1'C\x0eT)\x1by\xed\xb0\x8ca\x8e\xca\x91Uk)\xde\xb1\xde\xc8\x92\xc1L\xdb4\x8a\xbbP>\xd5{\x84\x85W\xec\x02\xd1Z\xfd2\xff\xb8{:\xcc\x07\x97E!\x9b\xbc\xfd8~9\xb84\x95\x01\xa7\xe4\xb2\xcf\x92\xbb\xcb\x97E\xbe\xb8*\x86\xf3\xdb\xd5\xb7%WS\xeb\x90[\x83.-\xfeO\xef^o\x11|\x8d@%2([\x9e\x1e\x15\xa2\x81C\x8d\x16f\xb0k=h\x0fd\xa1\xe9\xa0R\x1b\x8cA\xd0ZI\x8c\x9e%\x0b \xe9`X\xb34;XKu\x98\xd0\xab8\xa8\x98\x1a\xd8!\xed\x0cJ\xd7\xd6\xd2!\xd0\xe16XI\xd7Y\x99*\xf9\xd7\xae\x1f\x13\xc3\x03y\x91\xe02\xc8\x9e\x0c\x89no\xa8\xe4N\x0cL\x08\x1dX)2\x18\x8c\x08\x08n\xabm\xfc\xcf\xb1\x7f\x11\x95P\x11\x9fg|~\xf3\xac\x18fE>\x90\x9b\xff\x05\x07\x1bt[,\x03\x00\x00"
        reviews_extractor = ReviewsExtractor(**extractor_configs)
        extracted = reviews_extractor.extract_raw("")

        assert extracted == sample_extracted


def test_raw_extraction_process(extractor_configs, sample_extracted):
    """Test extracted and decompressed data is converted to a list."""
    with patch("bert_extractor.extractors.reviews.decompress") as decompress, patch(
        "requests.get"
    ) as response:
        response.return_value.content = b""
        decompress.return_value = b'{"overall": 5.0, "verified": true, "reviewTime": "09 1, 2016", "reviewerID": "A3CIUOJXQ5VDQ2", "asin": "B0000530HU", "style": {"Size:": " 7.0 oz", "Flavor:": " Classic Ice Blue"}, "reviewerName": "Shelly F", "reviewText": "As advertised. Reasonably priced", "summary": "Five Stars", "unixReviewTime": 1472688000}\n{"overall": 5.0, "verified": true, "reviewTime": "11 14, 2013", "reviewerID": "A3H7T87S984REU", "asin": "B0000530HU", "style": {"Size:": " 7.0 oz", "Flavor:": " Classic Ice Blue"}, "reviewerName": "houserules18", "reviewText": "Like the oder and the feel when I put it on my face.  I have tried other brands but the reviews from people I know they prefer the oder of this brand. Not hard on the face when dry.  Does not leave dry skin.", "summary": "Good for the face", "unixReviewTime": 1384387200}'
        reviews_extractor = ReviewsExtractor(**extractor_configs)
        extracted = reviews_extractor.extract_raw("")

        assert extracted == sample_extracted


def test_preprocess(extractor_configs, sample_extracted, sample_preprocessed):
    """Test preprocess create two list one for text and other for labels."""
    reviews_extractor = ReviewsExtractor(**extractor_configs)
    preprocessed_data = reviews_extractor.preprocess(sample_extracted)

    assert len(preprocessed_data) == 2
    assert preprocessed_data == sample_preprocessed
