import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define search bounds
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initial uniform sampling to generate starting points
        init_points = np.random.uniform(lb, ub, size=(15, self.dim))
        
        best_solution = None
        best_value = float('inf')

        for point in init_points:
            if self.evaluations >= self.budget:
                break

            # Optimize using local optimizer starting from the initial point
            remaining_budget = self.budget - self.evaluations
            local_budget = int(max(5, 0.1 * remaining_budget))  # Dynamic budget allocation
            result = minimize(self.bounded_func(func, lb, ub), point, method='Nelder-Mead',
                              options={'maxfev': local_budget})  # Changed from static to dynamic

            # Count the number of function evaluations
            self.evaluations += result.nfev

            # Update best solution if found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Refine adjustment factor for tighter search bounds around best known solution
            adjustment_factor = 0.02  # Changed from 0.03 to 0.02
            lb = np.maximum(lb, best_solution - adjustment_factor * (ub - lb))
            ub = np.minimum(ub, best_solution + adjustment_factor * (ub - lb))

        return best_solution

    def bounded_func(self, func, lb, ub):
        def func_with_bounds(x):
            # Clip the solution to remain within bounds
            x_clipped = np.clip(x, lb, ub)
            return func(x_clipped)
        return func_with_bounds