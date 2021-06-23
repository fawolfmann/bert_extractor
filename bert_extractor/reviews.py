#  type: ignore

"""Reviews Data Extractor"""

from gzip import decompress
import json

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from bert_extractor.base import BaseBERTExtractor


class ReviewsExtractor(BaseBERTExtractor):
    def extract_raw(self, url: str) -> pd.DataFrame:
        """[summary]

        Parameters
        ----------
        url : str
            [description]

        Returns
        -------
        pd.DataFrame
            [description]
        """
        reviews_json = json.loads(
            "["
            + decompress(requests.get(url).content)
            .decode("utf-8")
            .replace("}\n{", "},{")
            + "]"
        )
        df = pd.DataFrame(reviews_json)

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """[summary]

        Parameters
        ----------
        df : pd.DataFrame
            [description]

        Returns
        -------
        pd.DataFrame
            [description]
        """
        df["sentence"] = df["summary"] + " : " + df["reviewText"]
        df = df[["overall", "sentence"]]
        df.dropna(inplace=True)
        labels = df["overall"].astype(int).values
        sentences = df["sentence"].values

        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        input_ids = []
        attention_masks = []

        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                padding="max_length",
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="np",  # Return numpy tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict["input_ids"])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict["attention_mask"])

        a = """
        (
            train_inputs,
            validation_inputs,
            train_labels,
            validation_labels,
            train_mask,
            validation_masks,
        ) = train_test_split(
            input_ids, labels, attention_masks, random_state=2020, test_size=0.2
        )
        """
        print(a)
        return train_test_split(
            input_ids, labels, attention_masks, random_state=2020, test_size=0.2
        )
