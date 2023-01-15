"""Tests for the large language model interface."""

import os
import shutil

import pytest

from llmclone import utils
from llmclone.envs import create_tasks
from llmclone.llm_interface import LargeLanguageModel, OpenAILLM, \
    _llm_response_to_plan, run_llm_planning
from llmclone.structs import LLMResponse


class _ParrotLLM(LargeLanguageModel):
    """LLM that repeats the prompt, roughly."""

    def get_id(self):
        return "dummy"

    def _sample_completions(self,
                            prompt,
                            temperature,
                            seed,
                            stop_token,
                            num_completions=1):
        responses = []
        prompt_info = {
            "temperature": temperature,
            "seed": seed,
            "num_completions": num_completions,
            "stop_token": stop_token,
        }
        for _ in range(num_completions):
            text = (f"Prompt was: {prompt}. Seed: {seed}. "
                    f"Temp: {temperature:.1f}.")
            tokens = [text]
            logprobs = [0.0]
            other_info = {"dummy": 0}
            response = LLMResponse(prompt, text, tokens, logprobs,
                                   prompt_info.copy(), other_info)
            responses.append(response)
        return responses


class _MockLLM(LargeLanguageModel):
    """LLM with settable responses."""

    def __init__(self):
        self.responses = []

    def get_id(self):
        responses = "-".join(self.responses)
        return f"dummy-{hash(responses)}"

    def _sample_completions(self,
                            prompt,
                            temperature,
                            seed,
                            stop_token,
                            num_completions=1):
        del prompt, temperature, seed, num_completions  # unused
        next_response = self.responses.pop(0)
        response = LLMResponse("", next_response, [], [], {}, {})
        return [response]

    def sample_completions(self,
                           prompt,
                           temperature,
                           seed,
                           stop_token,
                           num_completions=1,
                           disable_cache=False):
        # Always disable the cache for tests.
        del disable_cache
        return super().sample_completions(prompt,
                                          temperature,
                                          seed,
                                          stop_token,
                                          num_completions,
                                          disable_cache=True)


def test_large_language_model():
    """Tests for LargeLanguageModel()."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_flags({
        "llm_cache_dir": cache_dir,
        "llm_use_cache_only": False,
        "llm_max_total_tokens": 700
    })
    # Remove the fake cache dir in case it's lying around from old tests.
    shutil.rmtree(cache_dir, ignore_errors=True)
    # Query a dummy LLM.
    llm = _ParrotLLM()
    assert llm.get_id() == "dummy"
    responses = llm.sample_completions("Hello world!", 0.5, 123, "", 3)
    completions = [r.response_text for r in responses]
    expected_completion = "Prompt was: Hello world!. Seed: 123. Temp: 0.5."
    assert completions == [expected_completion] * 3
    # Query it again, covering the case where we load from disk.
    responses = llm.sample_completions("Hello world!", 0.5, 123, "", 3)
    completions = [r.response_text for r in responses]
    assert completions == [expected_completion] * 3
    # Query with temperature 0.
    responses = llm.sample_completions("Hello world!", 0.0, 123, "", 3)
    completions = [r.response_text for r in responses]
    expected_completion = "Prompt was: Hello world!. Seed: 123. Temp: 0.0."
    assert completions == [expected_completion] * 3
    # Clean up the cache dir.
    shutil.rmtree(cache_dir)
    # Test llm_use_cache_only.
    utils.reset_flags({
        "llm_cache_dir": cache_dir,
        "llm_use_cache_only": True,
        "llm_max_total_tokens": 700
    })
    with pytest.raises(ValueError) as e:
        completions = llm.sample_completions("Hello world!", 0.5, 123, "", 3)
    assert "No cached response found for LLM prompt." in str(e)


def test_openai_llm():
    """Tests for OpenAILLM()."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_flags({
        "llm_cache_dir": cache_dir,
        "llm_use_cache_only": False,
        "llm_max_total_tokens": 700
    })
    if "OPENAI_API_KEY" not in os.environ:  # pragma: no cover
        os.environ["OPENAI_API_KEY"] = "dummy API key"
    # Create an OpenAILLM with the curie model.
    llm = OpenAILLM("text-curie-001")
    assert llm.get_id() == "openai-text-curie-001"
    # Uncomment this to test manually, but do NOT uncomment in master, because
    # each query costs money.
    # completions = llm.sample_completions("Hello", 0.5, 123, "", 2)
    # assert len(completions) == 2
    # completions2 = llm.sample_completions("Hello", 0.5, 123, "", 2)
    # assert completions == completions2
    # shutil.rmtree(cache_dir)

    # Test _raw_to_llm_response().
    raw_response = {
        "text": "Hello world",
        "logprobs": {
            "tokens": ["Hello", "world"],
            "token_logprobs": [-1.0, -2.0]
        }
    }
    prompt = "Dummy prompt"
    temperature = 0.5
    seed = 123
    num_completions = 1
    stop_token = "Q:"
    llm_response = llm._raw_to_llm_response(  # pylint: disable=protected-access
        raw_response, prompt, temperature, seed, stop_token, num_completions)
    assert llm_response.prompt_text == "Dummy prompt"
    assert llm_response.response_text == "Hello world"
    assert llm_response.tokens == ["Hello", "world"]
    assert llm_response.token_logprobs == [-1.0, -2.0]
    assert llm_response.prompt_info["temperature"] == temperature
    assert llm_response.prompt_info["seed"] == seed
    assert llm_response.prompt_info["num_completions"] == num_completions
    assert llm_response.prompt_info["stop_token"] == stop_token


@pytest.mark.parametrize("env_name", ["pyperplan-blocks", "pyperplan-miconic"])
def test_llm_planning(env_name):
    """Tests LLM planning."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_flags({
        "llm_cache_dir": cache_dir,
        "llm_model_name": "code-davinci-002",  # should not matter for test
        "llm_use_cache_only": False,
    })
    llm = _MockLLM()
    tasks, _, _ = create_tasks(env_name, 1, 0, 0)
    task = tasks[0]
    ideal_plan, _ = utils.run_planning(task)
    llm.responses = ["\n".join(ideal_plan)]
    plan = run_llm_planning(task, llm, [(task, ideal_plan)])
    assert plan == ideal_plan


def test_llm_planning_failure_cases():
    """Tests failure cases of LLM planning."""
    cache_dir = "_fake_llm_cache_dir"
    utils.reset_flags({
        "llm_cache_dir": cache_dir,
        "llm_model_name": "code-davinci-002",  # should not matter for test
        "llm_use_cache_only": False,
    })
    llm = _MockLLM()
    tasks, _, _ = create_tasks("pyperplan-elevators", 1, 0, 0)
    task = tasks[0]
    ideal_plan = ['(move-up-fast fast0 n0 n8)', '(board p2 slow0-0 n2 n0 n1)', '(board p0 fast0 n8 n0 n1)', '(move-up-slow slow0-0 n2 n4)', '(move-down-fast fast0 n8 n4)', '(move-down-slow slow0-0 n4 n3)', '(board p1 slow0-0 n3 n1 n2)', '(move-up-slow slow0-0 n3 n4)', '(leave p1 slow0-0 n4 n2 n1)', '(move-down-slow slow0-0 n4 n1)', '(board p1 fast0 n4 n1 n2)', '(leave p2 slow0-0 n1 n1 n0)', '(leave p0 fast0 n4 n2 n1)', '(move-up-fast fast0 n4 n6)', '(leave p1 fast0 n6 n1 n0)']  # pylint: disable=line-too-long
    ideal_response = "\n".join(ideal_plan)

    # Test general approach failure.
    llm.responses = ["garbage Q:"]
    plan = run_llm_planning(task, llm, [(task, ideal_plan)])
    assert not plan

    # Test failure cases of _llm_response_to_plan().
    assert _llm_response_to_plan(ideal_response, task)  # pylint: disable=protected-access
    # Cases where a line contains malformed parentheses.
    response = "()\n" + ideal_response  # should be skipped
    plan = _llm_response_to_plan(response, task)  # pylint: disable=protected-access
    assert len(plan) == len(ideal_plan)
    response = ")(\n" + ideal_response  # should not parse any plan
    plan = _llm_response_to_plan(response, task)  # pylint: disable=protected-access
    assert not plan
    # Case where there is an unmatched left parenthesis.
    response = ideal_response + "\n("  # should be skipped
    plan = _llm_response_to_plan(response, task)  # pylint: disable=protected-access
    assert len(plan) == len(ideal_plan)
    # Case where object names are incorrect.
    assert "(move-up-fast fast0 n0 n8)" in ideal_response
    response = ideal_response.replace("(move-up-fast fast0 n0 n8)",
                                      "(move-up-fast dummy n0 n8)")
    plan = _llm_response_to_plan(response, task)  # pylint: disable=protected-access
    assert len(plan) == len(ideal_plan) - 1
    # Case where operator names are incorrect.
    response = ideal_response.replace("(move-up-fast fast0 n0 n8)",
                                      "(dummy-move-up-fast fast0 n0 n8)")
    plan = _llm_response_to_plan(response, task)  # pylint: disable=protected-access
    assert len(plan) == len(ideal_plan) - 1
    # Cases where the type signature of the operator is wrong.
    response = ideal_response.replace("(move-up-fast fast0 n0 n8)",
                                      "(move-up-fast fast0 n0)")
    plan = _llm_response_to_plan(response, task)  # pylint: disable=protected-access
    assert len(plan) == len(ideal_plan) - 1
    response = ideal_response.replace("(move-up-fast fast0 n0 n8)",
                                      "(move-up-fast n0 fast0 n8)")
    plan = _llm_response_to_plan(response, task)  # pylint: disable=protected-access
    assert len(plan) == len(ideal_plan) - 1
    response = ideal_response.replace("(move-up-fast fast0 n0 n8)",
                                      "(move-up-fast fast0 n0 n8 n8)")
    plan = _llm_response_to_plan(response, task)  # pylint: disable=protected-access
    assert len(plan) == len(ideal_plan) - 1

    shutil.rmtree(cache_dir)
