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
        # Using BFGS as the local optimizer with multiple starts for robustness
        best_local_value = float('inf')
        best_local_solution = x0
        for _ in range(3):  # Run local optimization multiple times and take the best
            result = minimize(func, self.uniform_sample(bounds), method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
            if result.fun < best_local_value:
                best_local_value = result.fun
                best_local_solution = result.x
        return best_local_solution, best_local_value

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