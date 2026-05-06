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
# (these read run_history for fitness normalisation – no change needed)
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
    assert opt._effective_taboo_threshold(best_parent) == pytest.approx(0.0)


def test_threshold_full_for_worst_parent_with_scaling():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
    )
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 1", 1.0)]
    worst_parent = make_solution("c = 1", 0.0)
    assert opt._effective_taboo_threshold(worst_parent) == pytest.approx(0.4)


def test_threshold_half_for_midpoint_parent_with_scaling():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
    )
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 1", 1.0)]
    mid_parent = make_solution("c = 1", 0.5)
    assert opt._effective_taboo_threshold(mid_parent) == pytest.approx(0.2)


def test_threshold_minimization_inverts_scaling():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
        minimization=True,
    )
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 1", 1.0)]
    best_parent = make_solution("c = 1", 0.0)   # best for minimization
    worst_parent = make_solution("d = 1", 1.0)  # worst for minimization
    assert opt._effective_taboo_threshold(best_parent) == pytest.approx(0.0)
    assert opt._effective_taboo_threshold(worst_parent) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Unit tests for _is_taboo  (now compares against taboo_list)
# ---------------------------------------------------------------------------


def test_is_taboo_returns_false_when_taboo_list_empty():
    opt = _make_optimizer(taboo_mode=True, taboo_similarity_threshold=0.5)
    parent = make_solution("x = 1", 0.5)
    candidate = make_solution("x = 1", float("nan"))
    # taboo_list is empty by default
    assert opt._is_taboo(candidate, parent) is False


def test_is_taboo_returns_false_when_threshold_zero():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.4,
        taboo_fitness_scaling=True,
    )
    # run_history used for fitness normalisation
    opt.run_history = [make_solution("a = 1", 0.0), make_solution("b = 2", 1.0)]
    # taboo_list holds the actual entries to compare against
    opt.taboo_list = [make_solution("a = 1", 0.0)]
    best_parent = make_solution("c = 3", 1.0)  # norm=1 → effective threshold = 0
    candidate = make_solution("a = 1", float("nan"))
    # Threshold is 0 so nothing is taboo even if identical
    assert opt._is_taboo(candidate, best_parent) is False


def test_is_taboo_detects_identical_solution():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.5,
        taboo_fitness_scaling=False,
    )
    code = "x = 1"
    opt.taboo_list = [make_solution(code, 0.5)]
    candidate = make_solution(code, float("nan"))
    parent = make_solution("y = 2", 0.3)
    assert opt._is_taboo(candidate, parent) is True


def test_is_taboo_passes_distinct_solution():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_similarity_threshold=0.1,
        taboo_fitness_scaling=False,
    )
    opt.taboo_list = [make_solution("x = 1", 0.5)]
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
    opt.taboo_list = [make_solution("x = 1", 0.5)]
    candidate = make_solution("y = 2", float("nan"))
    parent = make_solution("z = 3", 0.3)

    result = opt._is_taboo(candidate, parent)

    assert result is True
    assert len(calls) >= 1


# ---------------------------------------------------------------------------
# Unit tests for _update_taboo_list strategies
# ---------------------------------------------------------------------------


def test_update_taboo_list_noop_when_taboo_mode_disabled():
    opt = _make_optimizer(taboo_mode=False)
    s = make_solution("x = 1", 0.5)
    opt._update_taboo_list([s])
    assert opt.taboo_list == []


def test_invalid_taboo_strategy_raises_valueerror():
    with pytest.raises(ValueError, match="taboo_strategy"):
        _make_optimizer(taboo_mode=True, taboo_strategy="unknown")


def test_update_taboo_list_always_adds_all_solutions():
    opt = _make_optimizer(taboo_mode=True, taboo_strategy="always")
    s1 = make_solution("a = 1", 0.9)
    s2 = make_solution("b = 2", 0.1)
    opt._update_taboo_list([s1, s2])
    assert s1 in opt.taboo_list
    assert s2 in opt.taboo_list


def test_update_taboo_list_poor_fitness_adds_bad_solutions_maximization():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_strategy="poor_fitness",
        taboo_poor_fitness_percentile=50.0,
    )
    # Build a run_history so we have a reference distribution
    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        opt.run_history.append(make_solution(f"x = {v}", v))

    bad = make_solution("bad = 0", 0.1)    # bottom 50 % → taboo'd
    good = make_solution("good = 0", 0.9)  # top 50 % → not taboo'd
    opt._update_taboo_list([bad, good])

    assert bad in opt.taboo_list
    assert good not in opt.taboo_list


def test_update_taboo_list_poor_fitness_adds_bad_solutions_minimization():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_strategy="poor_fitness",
        taboo_poor_fitness_percentile=50.0,
        minimization=True,
    )
    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        opt.run_history.append(make_solution(f"x = {v}", v))

    bad = make_solution("bad = 0", 0.9)    # high value is bad when minimizing
    good = make_solution("good = 0", 0.1)  # low value is good when minimizing
    opt._update_taboo_list([bad, good])

    assert bad in opt.taboo_list
    assert good not in opt.taboo_list


def test_update_taboo_list_poor_fitness_does_nothing_with_insufficient_history():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_strategy="poor_fitness",
        taboo_poor_fitness_percentile=25.0,
    )
    opt.run_history = [make_solution("a = 1", 0.5)]  # only 1 entry
    s = make_solution("b = 2", 0.1)
    opt._update_taboo_list([s])
    assert opt.taboo_list == []


def test_update_taboo_list_stagnation_does_not_add_on_first_call():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_strategy="stagnation",
        taboo_stagnation_window=3,
    )
    # Simulate best_so_far
    opt.best_so_far = make_solution("best", 0.5)
    s = make_solution("x = 1", 0.5)
    opt._update_taboo_list([s])  # first call → just records baseline
    assert opt.taboo_list == []
    assert opt._taboo_stagnation_counter == 0


def test_update_taboo_list_stagnation_adds_after_window():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_strategy="stagnation",
        taboo_stagnation_window=2,
    )
    opt.best_so_far = make_solution("best", 0.5)

    s = make_solution("x = 1", 0.5)
    opt._update_taboo_list([s])   # call 1: records baseline, counter stays 0

    # No improvement: same fitness
    opt._update_taboo_list([s])   # call 2: counter → 1 (< window=2, not yet)
    assert opt.taboo_list == []

    opt._update_taboo_list([s])   # call 3: counter → 2 (>= window) → adds
    assert s in opt.taboo_list


def test_update_taboo_list_stagnation_resets_counter_on_improvement():
    opt = _make_optimizer(
        taboo_mode=True,
        taboo_strategy="stagnation",
        taboo_stagnation_window=2,
    )
    opt.best_so_far = make_solution("best", 0.5)
    s = make_solution("x = 1", 0.5)

    opt._update_taboo_list([s])   # call 1: baseline recorded

    opt._update_taboo_list([s])   # call 2: no improvement → counter = 1

    # Simulate an improvement
    opt.best_so_far = make_solution("better", 0.9)
    opt._update_taboo_list([s])   # call 3: improvement detected → counter resets to 0
    assert opt._taboo_stagnation_counter == 0

    opt._update_taboo_list([s])   # call 4: no improvement → counter = 1 (< 2 → no add)
    assert opt.taboo_list == []


# ---------------------------------------------------------------------------
# Integration tests: retry behaviour in evolve_solution
# ---------------------------------------------------------------------------


def test_taboo_mode_retries_on_similar_solution():
    """When the first LLM response is taboo, evolve_solution retries."""

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
        taboo_max_retries=3,
        distance_metric=exact_distance,
    )

    seed = Solution(code=TABOO_CODE, name="Seed")
    seed.set_scores(0.5)
    opt.taboo_list = [seed]

    parent = Solution(code=TABOO_CODE, name="Parent")
    parent.set_scores(0.5)

    opt.llm.query = MagicMock(side_effect=[response_taboo, response_fresh])

    result = opt.evolve_solution(parent)

    assert result.name == "Fresh", f"Expected 'Fresh', got '{result.name}'"
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
    opt.taboo_list = [seed]

    parent = Solution(code=TABOO_CODE, name="Parent")
    parent.set_scores(0.5)

    opt.llm.query = MagicMock(return_value=response)

    result = opt.evolve_solution(parent)

    assert result.name == "Algo"
    assert opt.llm.query.call_count == 3  # max_retries + 1
