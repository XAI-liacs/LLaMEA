import os, sys, random
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pytest
from llamea import LLaMEA
from llamea.solution import Solution



class DummyLLM:
    def __init__(self):
        self.model = "DUMMY"

    def set_logger(self, logger):
        pass

    def sample_solution(self, *args, **kwargs):
        return Solution(name="dummy", code="")


def dummy_f(individual, logger=None):

    return individual


def make_algo_with_population(
    fitnesses, parent_selection="random", tournament_size=3, n_offspring=5
):
    algo = LLaMEA(
        f=dummy_f,
        llm=DummyLLM(),
        n_parents=len(fitnesses),
        n_offspring=n_offspring,
        parent_selection=parent_selection,
        tournament_size=tournament_size,
        log=False,
    )
    pop = []
    for i, fit in enumerate(fitnesses):
        s = Solution(name=str(i), code="")
        s.set_scores(fit)
        pop.append(s)
    algo.population = pop
    return algo


def test_tournament_selects_best():
    algo = make_algo_with_population(
        [1, 2, 3, 4], parent_selection="tournament", tournament_size=4, n_offspring=3
    )
    selected = algo._select_parents()

    assert len(selected) == 3
    assert all(s.fitness == 4 for s in selected)


def test_roulette_prefers_best():
    random.seed(0)
    np.random.seed(0)
    algo = make_algo_with_population(
        [1, 2, 3, 4], parent_selection="roulette", n_offspring=2
    )

    vals = []
    for _ in range(500):
        sel = algo._select_parents()
        vals.extend([s.fitness for s in sel])
    avg_sel = sum(vals) / len(vals)
    avg_pop = sum([1, 2, 3, 4]) / 4
    assert avg_sel > avg_pop


def test_roulette_minimization_prefers_lower():
    random.seed(1)
    np.random.seed(1)
    algo = make_algo_with_population(
        [10, 5, 1, 20], parent_selection="roulette", n_offspring=2
    )
    algo.minimization = True
    vals = []
    for _ in range(300):
        sel = algo._select_parents()
        vals.extend([s.fitness for s in sel])
    avg_sel = sum(vals) / len(vals)
    avg_pop = sum([10, 5, 1, 20]) / 4
    assert avg_sel < avg_pop


def test_random_selection_count():
    algo = make_algo_with_population(
        [1, 2, 3], parent_selection="random", n_offspring=7
    )
    selected = algo._select_parents()
    assert len(selected) == 7
