"""Data structures."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pyperplan.pddl.pddl import Action as _PyperplanAction
from pyperplan.pddl.pddl import Domain as _PyperplanDomain
from pyperplan.pddl.pddl import Predicate as _PyperplanPredicate
from pyperplan.pddl.pddl import Problem as _PyperplanProblem
from pyperplan.pddl.pddl import Type as _PyperplanType
from pyperplan.task import Task as _PyperplanTask

# Explicitly list the pyperplan data structures that we use.
PyperplanAction = _PyperplanAction
PyperplanDomain = _PyperplanDomain
PyperplanObject = str
PyperplanPredicate = _PyperplanPredicate
PyperplanProblem = _PyperplanProblem
PyperplanTask = _PyperplanTask
PyperplanType = _PyperplanType


@dataclass(frozen=True)
class Task:
    """A task is a PDDL domain file and problem file."""
    domain_file: Path
    problem_file: Path

    @cached_property
    def task_id(self) -> str:
        """A unique identifier for this task."""
        # Use the name of the domain from the domain file.
        domain_tag = "(domain "
        assert self.domain_str.count(domain_tag) == 1
        domain_tag_idx = self.domain_str.index(domain_tag)
        tag_close_rel_idx = self.domain_str[domain_tag_idx:].index(")")
        start = domain_tag_idx + len(domain_tag)
        end = domain_tag_idx + tag_close_rel_idx
        domain_name = self.domain_str[start:end].strip()
        assert domain_name
        # Use the problem filename, which is assumed unique within the domain.
        assert self.problem_file.name.endswith(".pddl")
        problem_name = self.problem_file.name[:-len(".pddl")]
        return f"{domain_name}__{problem_name}"

    @cached_property
    def problem_str(self) -> str:
        """Load and cache the problem string."""
        with open(self.problem_file, "r", encoding="utf-8") as f:
            problem_str = f.read()
        return problem_str

    @cached_property
    def domain_str(self) -> str:
        """Load and cache the domain string."""
        with open(self.domain_file, "r", encoding="utf-8") as f:
            domain_str = f.read()
        return domain_str
