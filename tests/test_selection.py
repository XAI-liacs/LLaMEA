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