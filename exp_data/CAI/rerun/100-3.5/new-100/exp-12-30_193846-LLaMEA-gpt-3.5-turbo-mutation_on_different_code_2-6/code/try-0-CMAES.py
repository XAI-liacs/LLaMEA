import numpy as np
import cma

class CMAES:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        es = cma.CMAEvolutionStrategy(np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim), 0.5)
        for _ in range(self.budget):
            solutions = es.ask()
            fitness_values = [func(x) for x in solutions]
            es.tell(solutions, fitness_values)
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self.f_opt:
                self.f_opt = fitness_values[best_idx]
                self.x_opt = solutions[best_idx]
        return self.f_opt, self.x_opt