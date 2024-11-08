import numpy as np
from scipy.optimize import differential_evolution

class HybridPSODEImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(x)

        bounds = np.array([(-5.0, 5.0)] * self.dim)  # Optimized bounds initialization
        popsize = 10
        num_particles = 5
        max_iter_de = 5

        def pso_de_optimizer():
            best_pos = differential_evolution(objective, bounds=bounds, maxiter=max_iter_de, popsize=popsize).x

            for _ in range(self.budget // num_particles - 1):
                pop = [best_pos + np.random.normal(0, 1, self.dim) for _ in range(num_particles)]
                for agent in pop:
                    de_result = differential_evolution(objective, bounds=bounds, maxiter=max_iter_de, popsize=popsize, init=agent)
                    if de_result.fun < objective(best_pos):
                        best_pos = de_result.x

            return best_pos

        return pso_de_optimizer()