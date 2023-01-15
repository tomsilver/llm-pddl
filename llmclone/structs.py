"""Data structures."""

from dataclasses import dataclass

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
    """A task is a PDDL domain str and problem str."""
    domain_str: str
    problem_str: str
