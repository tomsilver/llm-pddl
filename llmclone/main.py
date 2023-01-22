"""Main entry point for experiments."""

import logging
import sys
import time
from typing import List, Tuple

from pg3.policy_search import learn_policy
from pg3.utils import policy_satisfied

from llmclone import utils
from llmclone.envs import create_tasks
from llmclone.flags import FLAGS, parse_flags
from llmclone.llm_interface import OpenAILLM, run_llm_planning
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
    logging.info("Generating tasks.")
    prompt_tasks, train_tasks, eval_tasks = create_tasks(
        env_name=FLAGS.env,
        num_prompt=FLAGS.num_prompt_tasks,
        num_train=FLAGS.num_train_tasks,
        num_eval=FLAGS.num_eval_tasks,
    )

    # Create example plans for prompting.
    logging.info("Creating demos for prompting.")
    prompt_demos: List[Tuple[Task, Plan]] = []
    for task in prompt_tasks:
        plan, _ = utils.run_planning(task, planner=FLAGS.planner)
        assert plan is not None, "Planning failed"
        demo = (task, plan)
        prompt_demos.append(demo)

    # Get train and eval plans from LLM.
    logging.info("Querying LLM to get train and eval demos.")
    llm = OpenAILLM(FLAGS.llm_model_name)
    train_demos: List[Tuple[Task, Plan]] = []
    eval_demos: List[Tuple[Task, Plan]] = []
    for task_list, demo_list in [(train_tasks, train_demos),
                                 (eval_tasks, eval_demos)]:
        for task in task_list:
            plan = run_llm_planning(task, llm, prompt_demos)
            assert plan is not None, "LLM planning produced nothing"
            demo = (task, plan)
            demo_list.append(demo)

    # Use the LLM train plans to learn a policy by behavioral cloning.
    logging.info("Using train demos to learn a policy.")
    domain_str = train_demos[0][0].domain_str
    problem_strs = []
    demos = []
    for task, plan in train_demos:
        assert task.domain_str == domain_str
        problem_strs.append(task.problem_str)
        demos.append(plan)
    policy_str = learn_policy(
        domain_str,
        problem_strs,
        FLAGS.horizon,
        max_rule_params=FLAGS.pg3_max_rule_params,
        heuristic_name="demo_plan_comparison",
        demos=demos,
        search_method=FLAGS.pg3_search_method,
        gbfs_max_expansions=FLAGS.pg3_gbfs_max_expansions,
        hc_enforced_depth=FLAGS.pg3_hc_enforced_depth)
    logging.info(f"Learned policy:\n{policy_str}")

    # Evaluate the match between the policy and the LLM on the eval plans.
    logging.info("Evaluating the learned policy on the eval demos.")
    num_steps = 0
    num_matches = 0
    for task, plan in eval_demos:
        for a in plan:
            # Stop comparing if this action is not applicable.
            if not utils.action_is_valid_for_task(task, a):  # pragma: no cover
                break
            num_steps += 1
            # Check if this action matches the policy.
            match = policy_satisfied(policy_str, task.problem_str,
                                     task.domain_str, a)
            if match:
                num_matches += 1
            # Advance the task.
            task = utils.advance_task(task, a)
    acc = num_matches / num_steps
    logging.info(f"\nPolicy accuracy: {acc:.3f} ({num_matches}/{num_steps})")

    script_time = time.time() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


if __name__ == "__main__":  # pragma: no cover
    _main()
