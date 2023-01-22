"""Command line flags."""

import argparse
import logging

FLAGS = argparse.Namespace()  # set by parse_flags() below


def parse_flags() -> None:
    """Parse the command line flags and update global FLAGS."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num_prompt_tasks", default=2, type=int)
    parser.add_argument("--num_train_tasks", default=2, type=int)
    parser.add_argument("--num_eval_tasks", default=10, type=int)
    parser.add_argument("--planner", default="pyperplan")
    parser.add_argument("--llm_cache_dir", default="llm_cache", type=str)
    parser.add_argument("--llm_use_cache_only", action="store_true")
    parser.add_argument("--llm_model_name", default="code-davinci-002")
    parser.add_argument("--llm_max_total_tokens", default=4096, type=int)
    parser.add_argument("--horizon", default=100, type=int)
    parser.add_argument("--pg3_max_rule_params", default=8, type=int)
    parser.add_argument("--pg3_search_method",
                        default="hill_climbing",
                        type=str)
    parser.add_argument("--pg3_gbfs_max_expansions", default=100, type=int)
    parser.add_argument("--pg3_hc_enforced_depth", default=0, type=int)
    parser.add_argument('--debug',
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.INFO)
    args = parser.parse_args()
    FLAGS.__dict__.update(args.__dict__)
