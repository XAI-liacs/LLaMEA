import numpy as np
import cma

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        es = cma.CMAEvolutionStrategy(np.random.uniform(-5.0, 5.0, self.dim), 0.5)
        while not es.stop():
            solutions = es.ask()
            values = [func(x) for x in solutions]
            es.tell(solutions, values)
            self.f_opt = min(values)
            self.x_opt = solutions[np.argmin(values)]
        return self.f_opt, self.x_opt