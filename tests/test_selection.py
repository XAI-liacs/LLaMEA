from random import random

from llamea.llm import Dummy_LLM
from llamea import LLaMEA
from llamea.solution import Solution
from llamea.multi_objective_fitness import Fitness


def evaluate(solution: Solution, loggger) -> Solution:
    if random() < 0.1:
        # Simulate failure; or timeout
        solution.set_scores(float('nan'), "Runtime Error")
        return solution
    distance = 10 + (100 * random())
    fuel = 20 + (120 * random())

    fitness = Fitness({
        "Distance": distance,
        "Fuel": fuel
    })
    solution.set_scores(fitness, f"Got fitness {fitness}, best known distance is 9.6.")
    return solution

def dominates_minimisation(sol_a: Solution, sol_b: Solution) -> tuple[bool, bool]:
    better_in_all = True
    better_in_at_least_one = False

    for key in sol_a.fitness.keys():
        a_value = sol_a.fitness[key]
        b_value = sol_b.fitness[key]

        if a_value > b_value:
            better_in_all = False
        elif a_value < b_value:
            better_in_at_least_one = True

    return better_in_all, better_in_at_least_one


def test_run():
    llm = Dummy_LLM()
    llamea = LLaMEA(
        evaluate,
        llm,
        n_parents=10,
        n_offspring=10,
        multi_objective=True,
        multi_objective_keys=["Distance", "Fuel"],
        elitism=True,
        minimization=True,
        budget=100,
        experiment_name='multi_objective_test'
    )
    output : list[Solution] = llamea.run()
    for front in output:
        print(f"""
        {front.name}
        {front.code[:20]}
        {front.fitness}
        {front.feedback}
""")
    
    for i in range(len(output)-1):
        for j in range(i+1, len(output)):
            better_in_all, better_in_at_least_one = dominates_minimisation(output[i], output[j])
            assert not (better_in_all and better_in_at_least_one), f"Solution {i} should not dominate Solution {j} in same Pareto front."
    
    evaluations = [individual for individual in llamea.run_history if individual not in output]
    for eval_a in evaluations:
        for front in output:
            better_in_all, better_in_at_least_one = dominates_minimisation(eval_a, front)
            assert not (better_in_all and better_in_at_least_one), f"Evaluated solution {eval_a.fitness} should not dominate Pareto front solution {front.fitness}."


def _scored(name, a, b):
    solution = Solution(code=f"# {name}", name=name)
    solution.set_scores(Fitness({"a": a, "b": b}), "")
    return solution


def test_multi_objective_selection_maximisation_uses_maximisation_fronts():
    # Maximisation fronts: {A, B}, {C}, {D}. The minimisation fronts are
    # {B, D}, {C}, {A}, so reversing them would rank C (dominated by A)
    # ahead of B.
    for minimization, sign in [(False, 1.0), (True, -1.0)]:
        llamea = LLaMEA(
            evaluate,
            Dummy_LLM(),
            n_parents=2,
            n_offspring=2,
            multi_objective=True,
            multi_objective_keys=["a", "b"],
            minimization=minimization,
            log=False,
        )
        pool = [
            _scored(name, sign * a, sign * b)
            for name, a, b in [
                ("A", 2.0, 2.0),
                ("B", 0.0, 3.0),
                ("C", 1.0, 1.0),
                ("D", 0.5, 0.5),
            ]
        ]
        survivors = llamea.selection(parents=[], offspring=pool)
        assert sorted(s.name for s in survivors) == ["A", "B"]


def test_multi_objective_selection_keeps_n_parents():
    llamea = LLaMEA(
        evaluate,
        Dummy_LLM(),
        n_parents=2,
        n_offspring=4,
        multi_objective=True,
        multi_objective_keys=["a", "b"],
        minimization=True,
        log=False,
    )
    parents = [_scored("p0", 1.0, 5.0), _scored("p1", 5.0, 1.0)]
    offspring = [_scored(f"o{i}", 2.0 + i, 4.0 - i) for i in range(4)]
    assert len(llamea.selection(parents, offspring)) == 2
