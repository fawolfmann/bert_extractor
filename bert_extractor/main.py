"""Main file CLI for use this package, and usage example."""

import click

from bert_extractor.configs import read_config
from bert_extractor.constants import NER_KAGGLE_DATASET, REVIEWS_DATASET
from bert_extractor.extractors import BaseBERTExtractor, NERExtractor, ReviewsExtractor
from bert_extractor.utils import store_tensor


@click.command()
@click.option(
    "--config_path",
    type=click.STRING,
    help="Path to config file",
    default="./config/config_sample_reviews.json",
)
@click.option(
    "--output_path", type=click.STRING, default="./data/", help="Path to output file"
)
def main(config_path: str, output_path: str):
    """Main function to implement Bert Extractors.

    Parameters
    ----------
    config_path : str
        path to the configuration file.
    output_path : str
        path to where store the output.
    """
    extractor: BaseBERTExtractor
    configs = read_config(config_path)
    url = ""

    if configs["extractor_type"] == "reviews":
        extractor = ReviewsExtractor(**configs["extractor_config"])
        url = REVIEWS_DATASET.get(configs["extractor_url"])

    elif configs["extractor_type"] == "ner":
        extractor = NERExtractor(**configs["extractor_config"])
        url = NER_KAGGLE_DATASET.get(configs["extractor_url"])

    tensor = extractor.extract_preprocess(url)

    store_name = configs["extractor_type"] + "_" + url
    store_tensor(tensor, output_path, store_name)


if __name__ == "__main__":
    main()
