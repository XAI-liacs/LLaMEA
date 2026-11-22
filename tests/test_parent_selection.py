"""Regression tests for parent selection with multi-parent (crossover) operators.

`_select_parents` used to return a NumPy object array for ``"random"``
selection, so ``_select_parents(...) + [individual]`` broadcast ``+`` over
`Solution` objects and crashed every crossover operator.
"""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from llamea import Dummy_LLM, LLaMEA, Operator, Solution

METHODS = ["random", "roulette", "tournament"]


def _evaluate(solution: Solution, logger=None) -> Solution:
    solution.set_scores(random.random(), "ok")
    return solution


def _make_population(n=4):
    population = []
    for i in range(n):
        s = Solution(code=f"# parent-{i}\n", name=f"Parent{i}", description=f"p{i}")
        s.set_scores(float(i), f"feedback {i}")
        population.append(s)
    return population


def _make_llamea(method, operators=None, **kwargs):
    if operators is None:
        operators = [
            Operator(
                prompt="Combine these.",
                name="crossover",
                weight=1.0,
                number_of_parents=2,
            )
        ]
    params = dict(n_parents=4, n_offspring=2, log=False, budget=10)
    params.update(kwargs)
    return LLaMEA(
        _evaluate,
        Dummy_LLM(),
        operators=operators,
        parent_selection=method,
        **params,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("count", [0, 1, 5])
def test_select_parents_returns_list_of_requested_length(method, count):
    llamea = _make_llamea(method)
    llamea.population = _make_population()

    parents = llamea._select_parents(count=count)

    assert type(parents) is list
    assert len(parents) == count
    assert all(isinstance(p, Solution) for p in parents)
    assert all(any(p is q for q in llamea.population) for p in parents)
    # Concatenation with a list must work (this is what crossover does).
    extra = Solution(name="extra")
    combined = parents + [extra]
    assert len(combined) == count + 1 and combined[-1] is extra


def test_select_parents_defaults_to_n_offspring():
    llamea = _make_llamea("random")
    llamea.population = _make_population()
    assert len(llamea._select_parents()) == llamea.n_offspring


def test_random_selection_probe_returns_list():
    """The check BLADE uses to detect the bug must pass."""
    probe = SimpleNamespace(
        population=["a", "b"], parent_selection="random", n_offspring=1
    )
    parents = LLaMEA._select_parents(probe, count=1)
    assert isinstance(parents, list)
    assert parents + ["c"] == [parents[0], "c"]


def test_random_selection_samples_with_replacement():
    llamea = _make_llamea("random")
    llamea.population = _make_population(n=1)
    parents = llamea._select_parents(count=3)
    assert len(parents) == 3
    assert all(p is llamea.population[0] for p in parents)


@pytest.mark.parametrize("method", METHODS)
def test_evolve_solution_with_crossover_prompts_two_parents(method, monkeypatch):
    llamea = _make_llamea(method)
    llamea.population = _make_population()
    operator = llamea.operators[0]

    captured = {}
    original_query = llamea.llm.query

    def capturing_query(messages):
        captured["prompt"] = "\n".join(m["content"] for m in messages)
        return original_query(messages)

    monkeypatch.setattr(llamea.llm, "query", capturing_query)

    individual = llamea.population[0]
    offspring = llamea._evolve_solution(individual, operator)

    prompt = captured["prompt"]
    # Exactly two parent blocks (they may be the same solution, since random
    # selection samples with replacement).
    assert prompt.count("## With Code") == 2
    assert "# 1\n" in prompt and "# 2\n" in prompt
    assert prompt.count("# parent-") == 2
    assert len(offspring.parent_ids) == 2
    assert individual.code in prompt
    # Lineage points at real, logged solutions: the original individual (not
    # its throwaway copy) plus a member of the population.
    population_ids = {p.id for p in llamea.population}
    assert offspring.parent_ids[-1] == individual.id
    assert all(pid in population_ids for pid in offspring.parent_ids)
    assert "An exception occurred" not in (offspring.feedback or "")


@pytest.mark.parametrize("method", METHODS)
def test_construct_prompt_receives_two_parents(method, monkeypatch):
    llamea = _make_llamea(method)
    llamea.population = _make_population()
    operator = llamea.operators[0]

    seen = {}
    original = llamea.construct_prompt

    def spy(individuals, op):
        seen["individuals"] = individuals
        return original(individuals, op)

    monkeypatch.setattr(llamea, "construct_prompt", spy)
    llamea._evolve_solution(llamea.population[1], operator)

    assert type(seen["individuals"]) is list
    assert len(seen["individuals"]) == 2
    assert all(isinstance(p, Solution) for p in seen["individuals"])


def test_end_to_end_random_selection_with_crossover():
    random.seed(0)
    np.random.seed(0)
    mutation = Operator(
        prompt="Refine it.", name="mutation", weight=1e-9, number_of_parents=1
    )
    crossover = Operator(
        prompt="Combine these.", name="crossover", weight=1.0, number_of_parents=2
    )
    llamea = _make_llamea(
        "random",
        operators=[mutation, crossover],
        n_parents=2,
        n_offspring=2,
        budget=6,
        max_workers=1,
        parallel_backend="threading",
    )

    used = []
    original = llamea._evolve_solution

    def tracking(individual, operator):
        used.append(operator.name)
        return original(individual, operator)

    llamea._evolve_solution = tracking
    llamea.run()

    assert "crossover" in used
    assert len(llamea.run_history) >= 6
    for s in llamea.run_history:
        assert "An exception occurred" not in (s.feedback or "")
