import os
import random
from typing import Optional
from dataclasses import dataclass

from llamea import LLaMEA
from llamea import Solution
from llamea.llm import Ollama_LLM, Gemini_LLM
from llamea.pareto_archive import ParetoArchive
from llamea.loggers import ExperimentLogger
from llamea.utils import prepare_namespace
from llamea.multi_objective_fitness import Fitness

@dataclass
class Location:
    id: int
    x: int
    y: int
    weight: int

    def vectorise(self):
        return [self.id, self.x, self.y, self.weight]
    
    def __repr__(self):
        return f"Location(id: {self.id}, coordinates: ({self.x}, {self.y}), weight: {self.weight})"


def generate_tsp_test(seed: Optional[int] = None, size: int = 10):
    if seed is not None:
        random.seed(seed)
    depot = Location(0, 50, 50, 0)
    customers : list[Location] = []         # x, y, wight
    for id in range(size):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        weight = random.randint(10, 35)
        customers.append(Location(id + 1, x, y, weight))
    return depot, customers

depot, customers = generate_tsp_test(seed=69, size=32)

referable_dict = {}
referable_dict[0] = depot

for customer in customers:
    referable_dict[customer.id] = customer

def evaluate(solution: Solution, explogger: Optional[ExperimentLogger] = None):
    code = solution.code

    global_ns, issues = prepare_namespace(
        code,
        ['numpy', 'pymoo', 'typing'],
        explogger
    )
    local_ns = {}

    global_ns['Location'] = Location

    feedback = ""
    if issues:
        feedback += f"Import issues: {issues}. "
        print(f"Potential Issues {issues}.")

    compiled = compile(code, "<llm_code>", "exec")
    exec(compiled, global_ns, local_ns)

    cls = local_ns[solution.name]
    try:
        path_index = cls(
            depot.vectorise(),
            [customer.vectorise() for customer in customers]
        )()

    except Exception as e:
        solution.set_scores(
            Fitness({"Distance": float('inf'), "Fuel": float('inf')}),
            feedback=f"Runtime error: {e}",
            error=e
        )
        return solution

    # ---- Validate output ----
    if not isinstance(path_index, (list, tuple)):
        solution.set_scores(
            Fitness({"Distance": float('inf'), "Fuel": float('inf')}),
            feedback="Solver did not return a list of indices"
        )
        return solution

    if len(path_index) != len(customers):
        solution.set_scores(
            Fitness({"Distance": float('inf'), "Fuel": float('inf')}),
            feedback="Path length does not match number of customers"
        )
        return solution

    if len(set(path_index)) != len(path_index):
        solution.set_scores(
            Fitness({"Distance": float('inf'), "Fuel": float('inf')}),
            feedback="Path contains duplicate customer indices"
        )
        return solution

    if not all(idx in referable_dict for idx in path_index):
        solution.set_scores(
            Fitness({"Distance": float('inf'), "Fuel": float('inf')}),
            feedback="Path contains invalid customer indices"
        )
        return solution

    print(f"Path Index returned by LLM program: {path_index}")

    path: list[Location] = [referable_dict[idx] for idx in path_index]

    distance = 0.0
    previous = depot

    for current in path + [depot]:
        dx = current.x - previous.x
        dy = current.y - previous.y
        distance += (dx * dx + dy * dy) ** 0.5
        previous = current

    remaining = sum(c.weight for c in path)
    capacity = 1.1 * remaining

    fuel = 0.0
    previous = depot

    for current in path + [depot]:
        dx = current.x - previous.x
        dy = current.y - previous.y
        dist = (dx * dx + dy * dy) ** 0.5

        consumption_rate = 1.0 + (remaining / capacity)
        fuel += dist * consumption_rate

        remaining -= current.weight
        previous = current

    fitness = Fitness({
        "Distance": distance,
        "Fuel": fuel
    })

    solution.set_scores(
        fitness,
        feedback=f"Fitness {fitness} for path {path_index}"
    )
    return solution

if __name__ == "__main__":
    # key = os.getenv("GOOGLE_API__KEY")
    llm = Ollama_LLM("gemma3:12b")
    # llm = Gemini_LLM(key)

    role_prompt = "You are an excellent Scientific Programmer, who can write novel solution to solve optimisation problem."

    task_prompt = """Write a novel solution, for solving multi-objective (Distance, Fuel) Travelling Salesman Problem.
The salesman starts and ends at the depot, and he visits each customer only once.
Write a class with __init__ method that excepts a two parameters.
    * The first one is the depot, which is of type tuple(int, int, int, int); corresponding to its id, x-coordinate, y-coordinate, weight.
    * The second is customers which is a `list[tuple(int, int, int, int)]`, same corresponding values for the tuple.
        * So the class should instantiate as `__init__(depot: tuple[int, int, int, int], customers: list[tuple[int, int, int, int]])`.
    * The class should also have a `__call__()` method, that returns the path as a list of customer ids: `list[int]`.
        * `Note`: The returned list must not contain depot's id, it is accounted for by the evaluator.
"""
    example_prompt = """
An example program of this solution will be:
import random
class Multi_Objective_TSP:
    def __init__(depot, customer):
        self.depot = depot
        self.cusotmers = customers

    def __call__():
        customer_ids = [customer[0] for customer in customers]
        random.shuffle(customer_ids)
        return customer_ids
"""

    llamea_inst = LLaMEA(f=evaluate, 
           llm=llm,
           multi_objective=True,
           max_workers=3,
           n_offspring=3,
           n_parents=3,
           multi_objective_keys=['Distance', 'Fuel'],
           role_prompt=role_prompt,
           task_prompt=task_prompt,
           example_prompt=example_prompt,
            experiment_name="MOO-TSP",
            minimization=True,
            budget=27
           )

    solutions = llamea_inst.run()
    if isinstance(solutions, ParetoArchive):
        solutions = solutions.get_best()

    import matplotlib.pyplot as plt
    for index, solution in enumerate(solutions):
        print(index + 1)
        print(solutions.name)
        print(solutions.description)
        print(solutions.code)
        print(solutions.fitness)
        print("------------------------------------------------------------------------------------------------------------------------")