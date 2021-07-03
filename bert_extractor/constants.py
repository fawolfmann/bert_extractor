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
NER_KAGGLE_DATASET = {"conll_2003": "alaakhaled/conll003-englishversion"}

SPECIAL_TOKEN_LABEL = -100


# AMAZON REVIEWS
REVIEWS_DATASET = {
    "fashion": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz",
    "beauty": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty_5.json.gz",
    "appliances": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Appliances_5.json.gz",
}

# Configs

NER_CONFIG_TYPE = "ner"
REVIEWS_CONFIG_TYPE = "reviews"
KNOWN_CONFIGS_TYPES = [NER_CONFIG_TYPE, REVIEWS_CONFIG_TYPE]

KNOWN_NER_URLS = list(NER_KAGGLE_DATASET.keys())
KNOWN_REVIEWS_URLS = list(REVIEWS_DATASET.keys())
