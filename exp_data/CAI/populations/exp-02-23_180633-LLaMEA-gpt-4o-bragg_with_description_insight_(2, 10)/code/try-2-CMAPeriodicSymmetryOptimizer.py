import numpy as np
from cma import CMAEvolutionStrategy

class CMAPeriodicSymmetryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.sigma = 0.5  # Step size
        self.lambda_ = 10 + 3 * np.log(dim)  # Population size
        self.func_evals = 0

    def enforce_periodic_symmetry(self, x, lb, ub):
        # Ensure symmetry in the solution
        for i in range(0, self.dim, 2):
            period_segment = (ub - lb) / (self.dim // 2)
            x[i:i+2] = lb + (i % 2) * period_segment
        return np.clip(x, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x0 = np.random.uniform(lb, ub, self.dim)
        es = CMAEvolutionStrategy(x0, self.sigma, {'popsize': self.lambda_})

        while not es.stop() and self.func_evals < self.budget:
            solutions = es.ask()
            fitnesses = []

            for x in solutions:
                if self.func_evals >= self.budget:
                    break
                x = self.enforce_periodic_symmetry(x, lb, ub)
                fitness = -func(x)
                fitnesses.append(fitness)
                self.func_evals += 1

            es.tell(solutions, fitnesses)
            es.disp()

        result = es.result.xbest
        best_solution = self.enforce_periodic_symmetry(result, lb, ub)
        return best_solution