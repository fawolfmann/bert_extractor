# BERT Data Extractor
Python module to extract and preprocess data for BERT Classification.

In this package you will find to types of extractors, one for Text Classification and other for Token Classification, each of them is extracting an associated dataset.

BERT requires specific format as inputs. A BERT model needs a tokenized words, the tokenaztion dependes on which BERT model we are using, you can find all the BERT models [here](https://huggingface.co/transformers/pretrained_models.html), also all of this BERTs models can be use in this package setting it in the configuration file.

## Install
To install i used [Poetry](https://python-poetry.org/docs/) as environment isolation and dependencies solving tool.

To install Poetry download the source and run it:

install:
```
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

uninstall Poetry:
```
$ python get-poetry.py --uninstall
$ POETRY_UNINSTALL=1 python get-poetry.py
```

#### Install package

To install the package run:
```
$ poetry install
```

## Project structure
```
bert_extractor
├── bert_extractor:
│   ├── main: script that run the project, with CLI.
│   ├── configs: read and validate configurations.
│   ├── utils: utilities file to use in the package.
│   ├── constants: constants values.
│   └── extractors: bert_extractor python package.
│       ├── base: base class to BERT extractors.
│       ├── ner: NER sub class that extract and preprocess the data for Token Classification.
│       └── reviews: sub class extract and preprocess Amazon reviews for Text Classification.
│ 
├── config: folder with sample configuration files samples.
├── data: folder with extracted raw data samples.
└── tests: tests for all the package.
    └── extractor:
        ├── test_base_extractor: tests for base class.
        ├── test_ner_extractor: tests for ner class.
        ├── test_reviews_extractor: tests for reviews class.
        └── sample_data: examples of data to test.
```

## Run
For running purpose i created a [main.py](./bert_extractor/main.py) file that instantiate the extractors and execute them depending the configuration you set. It is also executable with CLI for that reason i use [Click](https://click.palletsprojects.com/en/8.0.x/) package.

### How to run it
Example command to run this:
```
$ poetry run main.py --config_path=../config/config_sample_ner.json --output_path=../data/
```

#### NER Dataset
The NER dataset is a CoNLL 2003 problem (Token classification) its from Kaggle so i use Kaggle's API to download the dataset. But to pass the credentials at runtime I fork the [repo](https://github.com/fawolfmann/kaggle-api) so it don't authenticate when you import the package.
To use it you have to set the [credentials](https://www.kaggle.com/docs/api#authentication) in the configuration file. Also i have a cached dataset on [data](./data) folder.


#### Amazon Reviews Dataset
The reviews dataset is public but you have to require access in a google form. In the web page there is a light dataset to use in development time, also i cached one dataset so you can try it.

This is a Text classification problem.

## Extractors

I create a base class for extraction and use inheritance for the specifics extractors.

I thought this problem as a data pipeline so for that reason i created the methods for each step:
- Extraction raw data.
- Preprocess raw data.
- BERT tokenization
    - Labels tokenization (if needed).
- Save tokenized output.

## Testing
For testing i use pytests, pytests sit on top of unitests and add some capabilities like fixtures and easier test creation process.

This repo has a [Github action](.github/workflows/ci.yml) that execute the tests in the [tests](./tests) folder on each commit in a PR o merging process, they run with [nox](https://nox.thea.codes/en/stable/), nox create a virtual environment install the package and run all the tests.


## Development
For this module I use this tools to lint code with coding good practice.
- black : code formatter.
- isort : sort imports.
- pylint : static code analysis tool.
- pre-commit : to run all the linting process before commit.

As the tests linting runs on [Github Actions](.github/workflows/ci.yml).

### GitFlow
For each new feature i create a new branch in the project, create a pull request and when i think i finish i merge the PR. This help me to work in different parts of the project at the same time.

## Production
For production i would suggest to use a data pipeline tools such as Airflow, Prefect, Dagster or any other. This tools help you maintain the data pipeline, knowing when a process fails, add retrying tools, you also can use them sensor, and many more.

### Pin versions
When going to production it is desirable to pin the exacts version of your package dependencies because you don't want any update of a dependencies can break your package.

### Poetry to production
The best way to go in production in my opinion y on a isolated environment, managed with docker or any other virtual environment tool.

#### To Docker

I add a [Dockerfile](Dockerfile) in this repo as an example on how i would run this on production.

#### To pip
Also if the requirements are others you can export this package to install it with pip. 

with this command you create a `requirements.txt`:
```
$ poetry export -f requirements.txt --output requirements.txt
```
after that you can run :

```
$ pip install -r requirements.txt
```

### Next Steps

Note: This package is in WIP there are some things in the code to improve, you will find the TODOs in the code.

- Implement with Airflow
- Parallelize with Spark if the volume of data requires.