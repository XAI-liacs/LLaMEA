from random import random

from llamea.llm import Dummy_LLM
from llamea import LLaMEA
from llamea.solution import Solution
from llamea.multi_objective_fitness import Fitness


def evaluate(solution: Solution, loggger) -> Solution:

    distance = 10 + (100 * random())
    fuel = 20 + (120 * random())

    fitness = Fitness({
        "Distance": distance,
        "Fuel": fuel
    })

    solution.set_scores(fitness, f"Got fitness {fitness}, best known distance is 9.6.")
    return solution

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
        minimization=True
    )
    output : list[Solution] = llamea.run()
    for front in output:
        print(f"""
        {front.name}
        {front.code[:20]}
        {front.fitness}
        {front.feedback}
""")
        print("------------------------------------------------------------------------------------")