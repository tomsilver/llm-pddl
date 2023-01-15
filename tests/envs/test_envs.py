"""Tests for task creation."""

import pytest

from llmclone.envs import create_tasks


def test_create_tasks():
    """Tests for create_tasks()."""

    # Test pyperplan task creation.
    prompt_tasks, train_tasks, eval_tasks = create_tasks(
        "pyperplan-blocks",
        num_prompt=1,
        num_train=2,
        num_eval=3,
    )
    assert len(prompt_tasks) == 1
    assert len(train_tasks) == 2
    assert len(eval_tasks) == 3

    # Test unknown task creation.
    with pytest.raises(NotImplementedError) as e:
        create_tasks("not a real env", 1, 1, 1)
    assert "Unrecognized env" in str(e)
