"""Create PDDL prompting, training, and evaluation tasks."""

import logging
from pathlib import Path
from typing import List, Tuple

from llmclone import utils
from llmclone.structs import Task


def create_tasks(
    env_name: str,
    num_prompt: int,
    num_train: int,
    num_eval: int,
) -> Tuple[List[Task], List[Task], List[Task]]:
    """Create PDDL prompting, training, and evaluation tasks."""
    total_num_tasks = num_prompt + num_train + num_eval

    if env_name.startswith("pyperplan-"):
        benchmark_name = env_name[len("pyperplan-"):]
        tasks = _get_pyperplan_tasks(benchmark_name, total_num_tasks)

    # Sort from smallest to largest.
    sorted_tasks = sorted(tasks, key=lambda t: t.size)
    # Use shortest for prompting, next shortest for training.
    prompt_tasks = sorted_tasks[:num_prompt]
    train_tasks = sorted_tasks[num_prompt:(num_prompt + num_train)]
    eval_tasks = sorted_tasks[(num_prompt + num_train):]
    assert len(eval_tasks) == num_eval

    return prompt_tasks, train_tasks, eval_tasks


def _get_pyperplan_tasks(benchmark_name: str, num_tasks: int) -> List[Task]:
    """Get PDDL tasks from the pyperplan benchmark set."""
    url_prefix = ("https://raw.githubusercontent.com/aibasel/pyperplan/main/"
                  f"benchmarks/{benchmark_name}")
    # Download the domain.
    domain_url = url_prefix + "/" + "domain.pddl"
    domain_str = utils.get_pddl_from_url(domain_url)
    # Download the problems.
    tasks = []
    for task_num in range(1, num_tasks + 1):
        problem_url = url_prefix + "/" + f"task{task_num:02d}.pddl"
        try:
            problem_str = utils.get_pddl_from_url(problem_url)
        except ValueError as e:
            logging.error(f"Could not download {problem_url}. Too many tasks?")
            raise e
        task = Task(domain_str, problem_str)
        tasks.append(task)
    return tasks
