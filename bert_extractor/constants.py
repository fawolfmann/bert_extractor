"""Constants file"""

# NER
NER_LABLES_MAP = {
    "B-LOC": 1,
    "B-PER": 2,
    "I-PER": 3,
    "I-LOC": 4,
    "B-MISC": 5,
    "I-MISC": 6,
    "B-ORG": 7,
    "I-ORG": 8,
    "O": 9,
}
NER_KAGGLE_DATASET = {"CoNLL003": "alaakhaled/conll003-englishversion"}

SPECIAL_TOKEN_LABEL = -100


# AMAZON REVIEWS
REVIEWS_DATASET = {
    "fashion": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz",
    "beauty": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty_5.json.gz",
    "appliances": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Appliances_5.json.gz",
}
