"""Tests for main.py."""

import sys

from llmclone.main import _main


def test_main():
    """Tests for main.py."""
    sys.argv = ["dummy", "--env", "pyperplan-blocks", "--seed", "123"]
    _main()  # should run
