import numpy as np
from scipy.optimize import differential_evolution

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(x)

        bounds = [(-5.0, 5.0)] * self.dim
        popsize = 10
        num_particles = 5

        def pso_de_optimizer():
            bounds_array = np.array(bounds)
            best_pos = differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=popsize).x

            for _ in range(self.budget // num_particles - 1):
                pop = [best_pos + np.random.normal(0, 1, self.dim) for _ in range(num_particles)]
                for agent in pop:
                    de_result = differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=popsize, init=agent)
                    if de_result.fun < objective(best_pos):
                        best_pos = de_result.x

            return best_pos

        return pso_de_optimizer()