"""Tests for the taboo search mode in LLaMEA."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from llamea.llamea import LLaMEA
from llamea.solution import Solution
from llamea.utils import code_distance


class DummyLLM:
    model = "dummy"

    def set_logger(self, logger):
        pass


def identity_f(individual, logger=None):
    return individual


def make_solution(code: str, fitness: float) -> Solution:
    s = Solution(code=code)
    s.set_scores(fitness)
    return s


def _make_optimizer(**kwargs):
    defaults = dict(f=identity_f, llm=DummyLLM(), log=False)
    defaults.update(kwargs)
    return LLaMEA(**defaults)


# ---------------------------------------------------------------------------
# Unit tests for _effective_taboo_threshold
# ---------------------------------------------------------------------------


def test_threshold_unchanged_when_fitness_scaling_disabled():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.2,
        taboo_fitness_scaling=False,
    )
    parent = make_solution("x = 1", 0.5)
    assert opt._effective_taboo_threshold(parent) == pytest.approx(0.2)


def test_threshold_unchanged_for_multi_objective():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.3,
        taboo_fitness_scaling=True,
        multi_objective=True,
        multi_objective_keys=["a"],
    )
    parent = make_solution("x = 1", 0.5)
    assert opt._effective_taboo_threshold(parent) == pytest.approx(0.3)


def test_threshold_unchanged_when_history_too_small():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.2,
        taboo_fitness_scaling=True,
    )
    # Only one solution in history (need at least 2 for scaling)
    opt.run_history = [make_solution("a = 1", 0.9)]
    parent = make_solution("b = 1", 0.5)
    assert opt._effective_taboo_threshold(parent) == pytest.approx(0.2)


def test_threshold_zero_for_best_parent_with_scaling():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
    )
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 1", 1.0)]
    best_parent = make_solution("c = 1", 1.0)
    # norm = (1.0 - 0.0) / (1.0 - 0.0) = 1.0 → threshold * (1 - 1) = 0
    assert opt._effective_taboo_threshold(best_parent) == pytest.approx(0.0)


def test_threshold_full_for_worst_parent_with_scaling():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
    )
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 1", 1.0)]
    worst_parent = make_solution("c = 1", 0.0)
    # norm = (0.0 - 0.0) / (1.0 - 0.0) = 0.0 → threshold * (1 - 0) = 0.4
    assert opt._effective_taboo_threshold(worst_parent) == pytest.approx(0.4)


def test_threshold_half_for_midpoint_parent_with_scaling():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
    )
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 1", 1.0)]
    mid_parent = make_solution("c = 1", 0.5)
    # norm = 0.5 → threshold * 0.5 = 0.2
    assert opt._effective_taboo_threshold(mid_parent) == pytest.approx(0.2)


def test_threshold_minimization_inverts_scaling():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
        minimization=True,
    )
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 1", 1.0)]
    best_parent = make_solution("c = 1", 0.0)  # best for minimization
    worst_parent = make_solution("d = 1", 1.0)  # worst for minimization
    # best parent (0.0 in minimization): norm → (1.0 - 0.0)/(1.0 - 0.0) = 1.0 → threshold→0
    assert opt._effective_taboo_threshold(best_parent) == pytest.approx(0.0)
    # worst parent (1.0 in minimization): norm = 0.0 → full threshold
    assert opt._effective_taboo_threshold(worst_parent) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Unit tests for _is_taboo
# ---------------------------------------------------------------------------


def test_is_taboo_returns_false_when_history_empty():
    opt = _make_optimizer(taboo_mode=True, taboo_similarity_threshold=0.5)
    parent = make_solution("x = 1", 0.5)
    candidate = make_solution("x = 1", float("nan"))
    assert opt._is_taboo(candidate, parent) is False


def test_is_taboo_returns_false_when_threshold_zero():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
    )
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 2", 1.0)]
    best_parent = make_solution("c = 3", 1.0)  # norm=1 → effective threshold = 0
    candidate = make_solution("a = 1", float("nan"))  # identical to history[0]
    # Even though identical, threshold is 0 so nothing is taboo
    assert opt._is_taboo(candidate, best_parent) is False


def test_is_taboo_detects_identical_solution():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.5,
        taboo_fitness_scaling=False,
    )
    code = "x = 1"
    opt.run_history = [make_solution(code, 0.5)]
    candidate = make_solution(code, float("nan"))
    parent = make_solution("y = 2", 0.3)
    assert opt._is_taboo(candidate, parent) is True


def test_is_taboo_passes_distinct_solution():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.1,
        taboo_fitness_scaling=False,
    )
    opt.run_history = [make_solution("x = 1", 0.5)]
    # A very different candidate
    candidate = make_solution(
        "import numpy as np\nclass Foo:\n    def bar(self):\n        return np.random.rand(10)",
        float("nan"),
    )
    parent = make_solution("y = 2", 0.3)
    assert opt._is_taboo(candidate, parent) is False


def test_is_taboo_uses_custom_distance_metric():
    calls = []

    def counting_metric(a, b):
        calls.append(1)
        return 0.0  # always identical → taboo

    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.5,
        taboo_fitness_scaling=False,
        distance_metric=counting_metric,
    )
    hist = make_solution("x = 1", 0.5)
    opt.run_history = [hist]
    candidate = make_solution("y = 2", float("nan"))
    parent = make_solution("z = 3", 0.3)

    result = opt._is_taboo(candidate, parent)

    assert result is True
    assert len(calls) >= 1  # metric was actually called


# ---------------------------------------------------------------------------
# Integration test: taboo mode prevents re-evaluating identical solutions
# ---------------------------------------------------------------------------


def test_taboo_mode_retries_on_similar_solution():
    """When the first LLM response is taboo, evolve_solution retries.

    We use a custom distance metric that classifies candidates by name so the
    test is independent of the AST-similarity computation.
    """

    TABOO_CODE = "class Taboo:\n    pass"
    FRESH_CODE = "class Fresh:\n    pass"

    response_taboo = (
        "# Description: taboo\n# Code:\n```python\n" + TABOO_CODE + "\n```"
    )
    response_fresh = (
        "# Description: fresh\n# Code:\n```python\n" + FRESH_CODE + "\n```"
    )

    eval_count = {"n": 0}

    def counting_f(ind, logger=None):
        eval_count["n"] += 1
        ind.set_scores(float(eval_count["n"]), "ok")
        return ind

    # Distance metric: 0.0 (identical) if both codes match, else 1.0 (distinct)
    def name_distance(a: Solution, b: Solution) -> float:
        return 0.0 if a.code == b.code else 1.0

    from llamea import Ollama_LLM

    opt = LLaMEA(
        counting_f,
        n_parents=1,
        n_offspring=1,
        llm=Ollama_LLM("model"),
        budget=2,
        log=False,
        taboo_mode=True,
        taboo_similarity_threshold=0.5,
        taboo_fitness_scaling=False,
        taboo_max_retries=3,
        distance_metric=name_distance,
    )

    # Pre-seed run_history with the taboo code
    seed = Solution(code=TABOO_CODE, name="Seed")
    seed.set_scores(0.5)
    opt.run_history = [seed]

    parent = Solution(code=TABOO_CODE, name="Parent")
    parent.set_scores(0.5)

    # LLM returns taboo code first, then fresh code
    opt.llm.query = MagicMock(side_effect=[response_taboo, response_fresh])

    result = opt.evolve_solution(parent)

    assert result.name == "Fresh", f"Expected 'Fresh', got '{result.name}'"
    # LLM must have been called twice: once returning taboo, once returning fresh
    assert opt.llm.query.call_count == 2


def test_taboo_mode_accepts_after_max_retries_exhausted():
    """If all retries return taboo solutions, the last one is still accepted."""

    TABOO_CODE = "class Algo:\n    pass"
    response = "# Description: dup\n# Code:\n```python\n" + TABOO_CODE + "\n```"

    eval_count = {"n": 0}

    def counting_f(ind, logger=None):
        eval_count["n"] += 1
        ind.set_scores(1.0, "ok")
        return ind

    def exact_distance(a: Solution, b: Solution) -> float:
        return 0.0 if a.code == b.code else 1.0

    from llamea import Ollama_LLM

    opt = LLaMEA(
        counting_f,
        n_parents=1,
        n_offspring=1,
        llm=Ollama_LLM("model"),
        budget=2,
        log=False,
        taboo_mode=True,
        taboo_similarity_threshold=0.5,
        taboo_fitness_scaling=False,
        taboo_max_retries=2,
        distance_metric=exact_distance,
    )

    seed = Solution(code=TABOO_CODE, name="Seed")
    seed.set_scores(0.5)
    opt.run_history = [seed]

    parent = Solution(code=TABOO_CODE, name="Parent")
    parent.set_scores(0.5)

    # Every LLM response is the taboo code (always rejected until retries exhausted)
    opt.llm.query = MagicMock(return_value=response)

    result = opt.evolve_solution(parent)

    # Despite being taboo, the candidate was accepted after retries exhausted
    assert result.name == "Algo"
    # LLM called max_retries + 1 = 3 times
    assert opt.llm.query.call_count == 3
