import numpy as np
from scipy.optimize import differential_evolution

class LeverageExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        self.f_best = np.inf
        self.step_size = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            # Explore
            x_explore = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            f_explore = func(x_explore)

            # Local search with adaptive step size
            step_size = self.step_size * (1 + np.random.uniform(-0.1, 0.1))
            x_local = self.x_best + step_size * np.random.uniform(-1, 1, size=self.dim)
            f_local = func(x_local)

            # Leverage the best exploration point
            if f_explore < f_local:
                self.x_best = x_explore
                self.f_best = f_explore

            # Differential evolution for local search
            bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
            x_dev = differential_evolution(func, bounds)
            f_dev = func(x_dev.x)

            # Update the best point if the local search is better
            if f_dev < self.f_best:
                self.x_best = x_dev.x
                self.f_best = f_dev

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

le = LeverageExploration(budget=100, dim=2)
le(func)
print("Best point:", le.x_best)
print("Best function value:", le.f_best)