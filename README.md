# Bert data Extractor
Python module to extract and preprocess data for Bert Classification

## Install
To install i used [Poetry](https://python-poetry.org/docs/) as environment and dependencies solving tool.

to install Poetry you have 2 method:

download the source and run it:

install:
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

uninstall Poetry:
python get-poetry.py --uninstall
POETRY_UNINSTALL=1 python get-poetry.py

with pipx: 

install:
pipx install poetry

uninstall:
pipx uninstall poetry

#### install project

To install the project run:
poetry install

## Run
For running purpose i created a main.py file that instantiate the extractors and execute them. Its also executable as CLI. 

### How to run it

poetry run main.py --option=[NER|reviews] --output=default.csv


## Testing
This repo has a Github action that execute the tests in the test folder, it run with [nox](https://nox.thea.codes/en/stable/).

## Development
For this module i used this tools to improve coding.
- black.
- isort.
- mypy.
- pylint.
- pre-commit.

also this runs on every commit.