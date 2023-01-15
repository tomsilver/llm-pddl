"""Main entry point for experiments."""

import logging
import sys
import time
from typing import List, Tuple

from llmclone import utils
from llmclone.envs import create_tasks
from llmclone.flags import FLAGS, parse_flags
from llmclone.structs import Plan, Task


def _main() -> None:
    # Basic setup.
    script_start = time.time()
    str_args = " ".join(sys.argv)
    # Parse command-line flags.
    parse_flags()
    # Set up logging.
    logging.basicConfig(level=FLAGS.loglevel,
                        format="%(message)s",
                        handlers=[logging.StreamHandler()])
    logging.info(f"Running command: python {str_args}")
    logging.info("Full config:")
    logging.info(FLAGS)
    logging.info(f"Git commit hash: {utils.get_git_commit_hash()}")

    # There are three sets of planning tasks: prompting, train, and eval.
    prompt_tasks, train_tasks, eval_tasks = create_tasks(
        env_name=FLAGS.env,
        num_prompt=FLAGS.num_prompt_tasks,
        num_train=FLAGS.num_train_tasks,
        num_eval=FLAGS.num_eval_tasks,
    )

    # Will use these in a forthcoming PR.
    del train_tasks, eval_tasks

    # Create example plans for prompting.
    prompt_demos: List[Tuple[Task, Plan]] = []
    for task in prompt_tasks:
        plan, _ = utils.run_planning(task, planner=FLAGS.planner)
        assert plan is not None, "Planning failed"
        demo = (task, plan)
        prompt_demos.append(demo)

    # Get train and eval plans from LLM.

    # Use the LLM train plans to learn a policy by behavioral cloning.

    # Evaluate the match between the policy and the LLM on the eval plans.

    script_time = time.time() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


if __name__ == "__main__":  # pragma: no cover
    _main()
