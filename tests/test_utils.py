"""Tests for utils.py."""

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmclone import utils
from llmclone.structs import Task


@pytest.fixture(scope="module", name="domain_str")
def _create_domain_str():
    domain_str = """;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 4 Op-blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (domain BLOCKS)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block)
	       (ontable ?x - block)
	       (clear ?x - block)
	       (handempty)
	       (holding ?x - block)
	       )
  (:action pick-up
	     :parameters (?x - block)
	     :precondition (and (clear ?x) (ontable ?x) (handempty))
	     :effect
	     (and (not (ontable ?x))
		   (not (clear ?x))
		   (not (handempty))
		   (holding ?x)))
  (:action put-down
	     :parameters (?x - block)
	     :precondition (holding ?x)
	     :effect
	     (and (not (holding ?x))
		   (clear ?x)
		   (handempty)
		   (ontable ?x)))
  (:action stack
	     :parameters (?x - block ?y - block)
	     :precondition (and (holding ?x) (clear ?y))
	     :effect
	     (and (not (holding ?x))
		   (not (clear ?y))
		   (clear ?x)
		   (handempty)
		   (on ?x ?y)))
  (:action unstack
	     :parameters (?x - block ?y - block)
	     :precondition (and (on ?x ?y) (clear ?x) (handempty))
	     :effect
	     (and (holding ?x)
		   (clear ?y)
		   (not (clear ?x))
		   (not (handempty))
		   (not (on ?x ?y)))))
"""
    return domain_str


@pytest.fixture(scope="module", name="problem_str")
def _create_problem_str():
    problem_str = """(define (problem blocks)
    (:domain blocks)
    (:objects
        d - block
        b - block
        a - block
        c - block
    )
    (:init
        (clear c)
        (clear b)
        (clear d)
        (ontable c)
        (ontable a)
        (ontable d)
        (on b a)
        (handempty)
    )
    (:goal (and (holding a)))
)
"""
    return problem_str


@pytest.fixture(scope="module", name="impossible_problem_str")
def _create_impossible_problem_str():
    problem_str = """(define (problem blocks)
    (:domain blocks)
    (:objects
        d - block
        b - block
        a - block
        c - block
    )
    (:init
        (clear c)
        (clear b)
        (clear d)
        (ontable c)
        (ontable d)
        (handempty)
    )
    (:goal (and (holding a)))
)
"""
    return problem_str


@patch('urllib.request.urlopen')
def test_get_pddl_from_url(mocked_urlopen, domain_str, problem_str):
    """Tests for get_pddl_from_url()."""
    today = date.today().strftime("%B %d, %Y")

    # Test getting domain files.
    cm = MagicMock()
    cm.read.return_value = domain_str.encode()
    cm.__enter__.return_value = cm
    mocked_urlopen.return_value = cm
    url = "https://not-a-real-pddl-repository.com/domain.pddl"
    expected_str = f"; Downloaded {today} from {url}\n" + domain_str.lower()
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        # Test getting a new domain file.
        pddl_str = utils.get_pddl_from_url(url, cache_dir)
        assert pddl_str == expected_str
        # Test loading an already downloaded domain file.
        pddl_str = utils.get_pddl_from_url(url, cache_dir)
        assert pddl_str == expected_str

    # Test getting problem files.
    cm.read.return_value = problem_str.encode()
    url = "https://not-a-real-pddl-repository.com/problem.pddl"
    expected_str = f"; Downloaded {today} from {url}\n" + problem_str.lower()
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        # Test getting a new problem file.
        pddl_str = utils.get_pddl_from_url(url, cache_dir)
        assert pddl_str == expected_str
        # Test loading an already downloaded problem file.
        pddl_str = utils.get_pddl_from_url(url, cache_dir)
        assert pddl_str == expected_str

    # Test getting a non-PDDL file, should fail.
    cm.read.return_value = "Not a real PDDL domain".encode()
    url = "https://not-a-real-pddl-repository.com/problem.pddl"
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        with pytest.raises(ValueError) as e:
            utils.get_pddl_from_url(url, cache_dir)
        assert f"PDDL file not found at {url}" in str(e)


def test_run_planning(domain_str, problem_str, impossible_problem_str):
    """Tests for run_planning().

    Fast downward is not tested because it's not easy to install on the
    github checks server.
    """
    # Test planning successfully.
    task = Task(domain_str, problem_str)
    plan, _ = utils.run_planning(task)
    assert plan is not None
    # Test planning in an impossible problem.
    impossible_task = Task(domain_str, impossible_problem_str)
    plan, _ = utils.run_planning(impossible_task)
    assert plan is None
    # Test planning with an invalid planner.
    with pytest.raises(NotImplementedError) as e:
        utils.run_planning(task, planner="not a real planner")
    assert "Unrecognized planner" in str(e)
