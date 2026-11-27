"""Multi-objective mode with an evaluator's own fitness class (issue #169).

``ForeignFitness`` behaves like BLADE's ``iohblade.fitness.Fitness``: the
same interface as ``llamea.Fitness``, but its comparisons return False for
any other class.
"""

import json

import pytest

from llamea import LLaMEA, Solution
from llamea.llm import LLM
from llamea.loggers import convert_to_serializable
from llamea.multi_objective_fitness import Fitness, is_multi_objective_fitness
from llamea.pareto_archive import ParetoArchive


class ForeignFitness:
    def __init__(self, value=None):
        self._fitness = dict(value or {})

    def keys(self):
        return self._fitness.keys()

    def __getitem__(self, key):
        return self._fitness.get(key, float("nan"))

    def to_vector(self):
        return [self._fitness[k] for k in sorted(self._fitness)]

    def to_dict(self):
        return dict(self._fitness)

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def _dom(self, other):
        be = all(self[k] <= other[k] for k in self.keys())
        sb = any(self[k] < other[k] for k in self.keys())
        return be, sb

    def __lt__(self, other):
        if not isinstance(other, ForeignFitness):
            return False
        be, sb = self._dom(other)
        return be and sb

    def __gt__(self, other):
        if not isinstance(other, ForeignFitness):
            return False
        be, sb = other._dom(self)
        return be and sb


class SequenceLLM(LLM):
    """Answers with failures (no code) first, then with code."""

    def __init__(self, failures=0):
        super().__init__("", "seq")
        self.failures = failures

    def query(self, session_messages):
        if self.failures > 0:
            self.failures -= 1
            return "no code here"
        return "# Description: X\n```python\nclass X:\n    pass\n```"


def foreign_eval(solution, logger=None):
    n = len(solution.id) % 7  # some spread between solutions
    solution.set_scores(
        ForeignFitness({"a": float(n), "b": float(7 - n)}), feedback="ok"
    )
    return solution


def _mo(llm, **kwargs):
    return LLaMEA(
        foreign_eval,
        llm,
        multi_objective=True,
        multi_objective_keys=["a", "b"],
        n_parents=2,
        n_offspring=2,
        budget=8,
        max_workers=1,
        log=False,
        **kwargs,
    )


def test_interface_check():
    assert is_multi_objective_fitness(Fitness({"a": 1.0}))
    assert is_multi_objective_fitness(ForeignFitness({"a": 1.0}))
    assert not is_multi_objective_fitness(1.0)
    assert not is_multi_objective_fitness({"a": 1.0})


def test_foreign_fitness_is_kept_not_replaced_by_worst():
    opt = _mo(SequenceLLM())
    s = Solution(code="x")
    s.set_scores(ForeignFitness({"a": 1.0, "b": 2.0}))
    (out,) = opt._ensure_fitness_evaluates([s])
    assert isinstance(out.fitness, ForeignFitness)
    assert out.fitness.to_dict() == {"a": 1.0, "b": 2.0}


def test_run_with_foreign_fitness_returns_a_front():
    front = _mo(SequenceLLM()).run()
    assert front
    assert all(isinstance(s.fitness, ForeignFitness) for s in front)
    assert all(v != float("-inf") for s in front for v in s.fitness.to_vector())


def test_placeholders_use_the_evaluators_class():
    # The first two LLM answers have no code, so the first generation holds
    # worst-value placeholders created before the evaluator's class is known.
    opt = _mo(SequenceLLM(failures=2))
    opt.run()
    for individual in opt.population + opt.run_history:
        assert isinstance(individual.fitness, ForeignFitness)
    placeholders = [
        s for s in opt.run_history if s.fitness.to_vector() == [float("-inf")] * 2
    ]
    assert placeholders  # the failures are still recorded as worst


def test_llamea_fitness_compares_with_foreign_fitness():
    better = Fitness({"a": 2.0, "b": 2.0})
    worse = ForeignFitness({"a": 1.0, "b": 1.0})
    assert better > worse
    assert not better < worse
    assert better >= worse
    assert Fitness({"a": 1.0, "b": 1.0}) == worse


def test_operator_rewards_with_foreign_fitness():
    opt = _mo(SequenceLLM())
    parent, child = Solution(code="p"), Solution(code="c")
    parent.set_scores(ForeignFitness({"a": 1.0, "b": 1.0}))
    child.set_scores(ForeignFitness({"a": 2.0, "b": 2.0}))
    assert opt._operator_rewards([parent], child) == 1  # maximisation


def test_pareto_archive_and_serialisation_accept_foreign_fitness():
    archive = ParetoArchive(minimisation=False)
    a, b = Solution(code="a"), Solution(code="b")
    a.set_scores(ForeignFitness({"a": 1.0, "b": 3.0}))
    b.set_scores(ForeignFitness({"a": 3.0, "b": 1.0}))
    archive.add_solutions([a, b])
    assert {s.id for s in archive.get_best()} == {a.id, b.id}
    assert a.get_fitness_vector() == [1.0, 3.0]
    data = convert_to_serializable({"fitness": a.fitness})
    assert json.loads(json.dumps(data)) == {"fitness": {"a": 1.0, "b": 3.0}}


def test_llamea_fitness_is_unhashable_but_equal_by_value():
    with pytest.raises(TypeError):
        hash(Fitness({"a": 1.0}))
    assert Fitness({"a": 1.0}) == Fitness({"a": 1.0})
