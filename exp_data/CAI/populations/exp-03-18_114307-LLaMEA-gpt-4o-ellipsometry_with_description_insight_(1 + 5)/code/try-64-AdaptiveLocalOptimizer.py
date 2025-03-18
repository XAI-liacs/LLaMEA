import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.memory = []  # Initialize memory to store historical best solutions

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        init_points = np.random.uniform(lb, ub, size=(15, self.dim))
        best_solution = None
        best_value = float('inf')

        for point in init_points:
            if self.evaluations >= self.budget:
                break

            result = minimize(self.bounded_func(func, lb, ub), point, method='Nelder-Mead',
                              options={'maxfev': self.budget - self.evaluations})
            
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                self.memory.append((best_solution, best_value))  # Store best solutions in memory

            adjustment_factor = 0.02 + 0.01*(self.evaluations/self.budget)
            lb = np.maximum(lb, best_solution - adjustment_factor * (ub - lb))
            ub = np.minimum(ub, best_solution + adjustment_factor * (ub - lb))

        # Incorporate memory into final decision making
        if self.memory:
            best_solution = min(self.memory, key=lambda x: x[1])[0]

        return best_solution

    def bounded_func(self, func, lb, ub):
        def func_with_bounds(x):
            x_clipped = np.clip(x, lb, ub)
            return func(x_clipped)
        return func_with_bounds