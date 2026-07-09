import random
from typing import Optional
from dataclasses import dataclass

from llamea.utils import prepare_namespace
from llamea import LLaMEA, Ollama_LLM, Operator, Fitness, Solution, ExperimentLogger


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
    """Generate a depot and customer set for the synthetic TSP task."""
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
    """Evaluate generated solver code on a two-objective TSP benchmark.

    The generated class must return a permutation of customer ids. The evaluator
    validates the route, computes total travel distance and load-dependent fuel
    usage, then stores a ``Fitness`` object with both objectives.
    """
    code = solution.code

    global_ns, issues = prepare_namespace(
        code,
        ['numpy', 'pymoo', 'typing', 'scipy'],
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
if __name__ == '__main__':

    llm = Ollama_LLM('gemma4:latest')

    operators = [
        Operator('mutation',
                 'Write a new algorithm, that applies small but significant changes to the mentioned algorithms',
                 0.2,
                 1
        ),
        Operator(
            'mutation',
            'Write a new algorithm that is completely different that the one mentioned.',
            0.2,
            1
        ),
        Operator(
            'crosover',
            'Write a new algorithm that is build from the strengths of a subset of the mentioned algorithms.',
            0.6,
            3
        )
    ]

    llamea = LLaMEA(
        evaluate,
        llm,
        5,
        5,
        operators=operators,
        parent_selection='tournament'
    )

    llamea.run()