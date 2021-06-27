# BERT data Extractor
Python module to extract and preprocess data for BERT Classification.
BERT have specific format requires...

## Install
To install i used [Poetry](https://python-poetry.org/docs/) as environment and dependencies solving tool.

To install Poetry you have 2 method:

download the source and run it:

install:
```
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

uninstall Poetry:
```
$ python get-poetry.py --uninstall
$ POETRY_UNINSTALL=1 python get-poetry.py
```

with pipx: 

install:
```
$ pipx install poetry
```

uninstall:
```
$ pipx uninstall poetry
```

#### Install project

To install the project run:
```
$ poetry install
```

## Project structure
bert_extractor
├── bert_extractor: bert_extractor python package
│   ├── base: base class to BERT extractors
│   ├── ner: NER sub class that extract and preprocess the data.
│   └── reviews: misc project utils
└── tests: python tests


## Run
For running purpose i created a main.py file that instantiate the extractors and execute them. Its also executable as CLI. 

### How to run it

```
$ poetry run main.py --option=[NER|reviews] --output=default.csv
```

## Testing
This repo has a Github action that execute the tests in the [tests](./tests) folder, it run with [nox](https://nox.thea.codes/en/stable/).

For testing i use pytests, pytests sit on top of unitests and add some capabilities like ...
for mocking requests i use [responses](https://github.com/getsentry/responses) package.

## Development
For this module I used this tools to lint code with coding good practice.
- black.
- isort.
- pylint.
- pre-commit.

As the tests linting runs on Github Actions.

## Productionable
### pin version
### poetry to prod
#### to pip
#### to docker