"""Data structures."""

from dataclasses import dataclass
from functools import cached_property

from pyperplan.pddl.parser import Parser
from pyperplan.pddl.pddl import Action as PyperplanAction
from pyperplan.pddl.pddl import Domain as PyperplanDomain
from pyperplan.pddl.pddl import Predicate as PyperplanPredicate
from pyperplan.pddl.pddl import Problem as PyperplanProblem
from pyperplan.pddl.pddl import Type as PyperplanType
from pyperplan.task import Task as PyperplanTask

PyperplanObject = str


@dataclass(frozen=True)
class Task:
    """A task is a PDDL domain str and problem str."""
    domain_str: str
    problem_str: str

    @cached_property
    def _parser(self) -> Parser:
        parser = Parser(None)
        parser.domInput = self.domain_str
        parser.probInput = self.problem_str
        return parser

    @cached_property
    def domain(self) -> PyperplanDomain:
        """The parsed PDDL domain for this task."""
        return self._parser.parse_domain(read_from_file=False)

    @cached_property
    def problem(self) -> PyperplanProblem:
        """The parsed PDDL problem for this task."""
        return self._parser.parse_problem(self.domain, read_from_file=False)

    @cached_property
    def size(self) -> int:
        """A crude measure of task complexity."""
        prob = self.problem
        return len(prob.objects) + len(prob.initial_state) + len(prob.goal)
