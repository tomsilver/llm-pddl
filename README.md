# LLM Behavioral Cloning

Under development.

## Requirements

- Python 3.8+
- Tested on MacOS Catalina

## Instructions For Contributing

### First Time

- (Highly recommended) Make a virtual environment:
  - `virtualenv venv`
  - `source venv/bin/activate`
- Clone this repository with submodules: `git clone https://github.com/tomsilver/llm-pddl`
- Run `pip install -e .[develop]` to install the main dependencies for development.

### Developing

- Before merging code, make sure you pass these checks:
  - `pytest -s tests/ --cov-config=.coveragerc --cov=llmclone/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered --durations=0`
  - `mypy .`
  - `pytest . --pylint -m pylint --pylint-rcfile=.llmclone_pylintrc`
  - `./run_autoformat.sh`
- The first one is the unit testing check, which verifies that unit tests pass and that code is adequately covered. The "100" means that all lines in every file must be covered.
- The second one is the static typing check, which uses Mypy to verify type annotations.
- The third one is the linter check, which runs Pylint with the custom config file `.llmclone_pylintrc` in the root of this repository. Feel free to edit this file as necessary.
- The fourth one is the autoformatting check, which uses the custom config files `.style.yapf` and `.isort.cfg` in the root of this repository.
