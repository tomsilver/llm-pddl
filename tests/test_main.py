"""Tests for main.py."""

import sys
from unittest.mock import MagicMock

from llmclone import utils

# Need to mock the LLM interface before importing _main().
fake_llm_plan = lambda task, _1, _2: utils.run_pyperplan_planning(task)[0]
mock = MagicMock()
mock.run_llm_planning.side_effect = fake_llm_plan
sys.modules['llmclone.llm_interface'] = mock

from llmclone.main import _main  # pylint: disable=wrong-import-position


def test_main():
    """Tests for main.py."""
    sys.argv = [
        "dummy", "--env", "pyperplan-blocks", "--seed", "123",
        "--num_prompt_tasks", "1", "--num_train_tasks", "1",
        "--num_eval_tasks", "1"
    ]
    _main()  # should run
