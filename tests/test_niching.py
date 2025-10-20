import random

import numpy as np
import pytest

from llamea.llamea import LLaMEA
from llamea.solution import Solution
from llamea.utils import code_distance


class DummyLLM:
    model = "dummy"


def identity_f(individual, logger=None):
    return individual


def make_solution(code: str, fitness: float) -> Solution:
    ind = Solution(code=code)
    ind.set_scores(fitness)
    return ind


def test_fitness_sharing_reduces_fitness_for_similar_individuals():
    optimizer = LLaMEA(
        f=identity_f,
        llm=DummyLLM(),
        niching="sharing",
        niche_radius=1.0,
        log=False,
    )
    s1 = make_solution("a=1", 10.0)
    s2 = make_solution("a=1", 20.0)
    optimizer.apply_niching([s1, s2])
    assert s1.fitness == pytest.approx(5.0)
    assert s2.fitness == pytest.approx(10.0)


def test_clearing_sets_inferior_individual_fitness_to_worst_value():
    optimizer = LLaMEA(
        f=identity_f,
        llm=DummyLLM(),
        niching="clearing",
        niche_radius=1.0,
        clearing_interval=1,
        log=False,
    )
    best = make_solution("a=1", 10.0)
    worst = make_solution("a=1", 5.0)
    optimizer.apply_niching([best, worst])
    assert best.fitness == pytest.approx(10.0)
    assert np.isneginf(worst.fitness)


def test_adaptive_niche_radius_updates_to_mean_distance():
    optimizer = LLaMEA(
        f=identity_f,
        llm=DummyLLM(),
        niching="sharing",
        niche_radius=0.5,
        adaptive_niche_radius=True,
        log=False,
    )
    s1 = make_solution("a=1", 1.0)
    s2 = make_solution("b=1", 1.0)
    expected = code_distance("a=1", "b=1")
    optimizer.apply_niching([s1, s2])
    assert optimizer.niche_radius == pytest.approx(expected)


def descriptor_by_code_length(solution: Solution) -> tuple[float, ...]:
    return (float(len(solution.code)),)


def test_map_elites_keeps_best_individual_per_cell():
    optimizer = LLaMEA(
        f=identity_f,
        llm=DummyLLM(),
        niching="map_elites",
        behavior_descriptor=descriptor_by_code_length,
        map_elites_bins=(3,),
        log=False,
    )
    weak = make_solution("ab", 1.0)
    strong = make_solution("ab", 5.0)
    elites = optimizer.apply_niching([weak, strong])

    assert len(optimizer.map_elites_archive) == 1
    assert optimizer.map_elites_archive[(0,)] is strong
    assert strong in elites


def test_map_elites_selection_draws_from_archive():
    random.seed(0)
    optimizer = LLaMEA(
        f=identity_f,
        llm=DummyLLM(),
        niching="map_elites",
        behavior_descriptor=descriptor_by_code_length,
        map_elites_bins=(3,),
        n_parents=2,
        elitism=True,
        log=False,
    )
    individuals = [
        make_solution("a", 1.0),
        make_solution("bb", 2.0),
        make_solution("ccc", 3.0),
    ]
    selected = optimizer.selection([], individuals)

    assert len(optimizer.map_elites_archive) == 3
    assert len(selected) == 2
    archive_values = set(optimizer.map_elites_archive.values())
    assert all(ind in archive_values for ind in selected)
