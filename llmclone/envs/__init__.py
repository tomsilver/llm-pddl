"""Create PDDL prompting, training, and evaluation tasks."""

from pathlib import Path
from typing import List, Tuple

from llmclone.structs import Task


def create_tasks(
    env_name: str,
    num_prompt: int,
    num_train: int,
    num_eval: int,
) -> Tuple[List[Task], List[Task], List[Task]]:
    """Create PDDL prompting, training, and evaluation tasks."""
    # Placeholder for future PR.
    del env_name
    dummy_task = Task(Path("dummy"), Path("dummy"))
    prompt_tasks = [dummy_task for _ in range(num_prompt)]
    train_tasks = [dummy_task for _ in range(num_train)]
    eval_tasks = [dummy_task for _ in range(num_eval)]
    return prompt_tasks, train_tasks, eval_tasks
