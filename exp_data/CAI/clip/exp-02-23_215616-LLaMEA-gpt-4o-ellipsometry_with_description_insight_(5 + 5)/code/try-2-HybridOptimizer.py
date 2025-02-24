import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def uniform_sample(self, bounds):
        return np.array([np.random.uniform(low, high) for low, high in zip(bounds.lb, bounds.ub)])

    def local_optimization(self, func, x0, bounds):
        # Using BFGS as the local optimizer
        result = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        best_solution = None
        best_value = float('inf')

        # Initial exploration phase using uniform sampling
        while self.evaluations < self.budget * 0.25:
            x0 = self.uniform_sample(bounds)
            value = func(x0)
            self.evaluations += 1

            if value < best_value:
                best_value = value
                best_solution = x0

        # Exploitation phase using local optimization
        while self.evaluations < self.budget:
            optimized_solution, optimized_value = self.local_optimization(func, best_solution, bounds)
            self.evaluations += 1  # Assuming one function evaluation for the local optimization step

            if optimized_value < best_value:
                best_value = optimized_value
                best_solution = optimized_solution

        return best_solution