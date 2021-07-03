"""Main file CLI for use this package, and usage example."""

import click

from bert_extractor.configs import read_config
from bert_extractor.extractors import BaseBERTExtractor, NERExtractor, ReviewsExtractor
from bert_extractor.utils import store_tensor


@click.command()
@click.option(
    "--config_path",
    type=click.STRING,
    help="Path to config file",
    default="./config/config_sample.json",
)
@click.option("--output_path", type=click.STRING, help="Path to output file")
@click.option("--output_format", type=click.STRING, help="type of file to save")
def main(config_path: str, output_path: str, output_format: str):
    """Main function to implement Bert Extractors.

    Parameters
    ----------
    config_path : str
        path to the configuration file.
    output_path : str
        path to where store the output.
    output_format : str
        desired output format option (pickle, zip, ...)
    """
    extractor: BaseBERTExtractor
    configs = read_config(config_path)

    if configs["extractor_type"] == "reviews":
        extractor = ReviewsExtractor(**configs["extractor_config"])
    elif configs["extractor_type"] == "ner":
        extractor = NERExtractor(**configs["extractor_config"])

    tensor = extractor.extract_preprocess(configs["extractor_url"])

    store_name = configs["extractor_type"] + configs["extractor_url"]
    store_tensor(tensor, output_path, store_name)


if __name__ == "__main__":
    main()
