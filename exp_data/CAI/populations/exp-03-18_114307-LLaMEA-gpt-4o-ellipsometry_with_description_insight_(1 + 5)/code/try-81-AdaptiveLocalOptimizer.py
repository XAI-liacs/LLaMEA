import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        init_points = np.random.uniform(lb, ub, size=(10, self.dim))  # Reduced initial points for diverse sampling
        
        best_solution = None
        best_value = float('inf')

        for point in init_points:
            if self.evaluations >= self.budget:
                break

            # Use L-BFGS-B for bounded optimization with local step-size adaptation
            result = minimize(self.bounded_func(func, lb, ub), point, method='L-BFGS-B',
                              options={'maxfun': self.budget - self.evaluations})
            
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Adaptive adjustment factor for refined bounds
            adjustment_factor = 0.05 + 0.01 * (self.evaluations / self.budget)  # Adjusted dynamic factor
            lb = np.maximum(lb, best_solution - adjustment_factor * (ub - lb))
            ub = np.minimum(ub, best_solution + adjustment_factor * (ub - lb))

        # Dynamic resampling if budget allows for exploration
        if self.evaluations < self.budget:
            self.dynamic_resampling(func, lb, ub)

        return best_solution

    def bounded_func(self, func, lb, ub):
        def func_with_bounds(x):
            x_clipped = np.clip(x, lb, ub)
            return func(x_clipped)
        return func_with_bounds

    def dynamic_resampling(self, func, lb, ub):
        # Additional exploration near the best solution
        resample_points = np.random.uniform(lb, ub, size=(5, self.dim))
        for point in resample_points:
            if self.evaluations >= self.budget:
                break
            result = minimize(self.bounded_func(func, lb, ub), point, method='Nelder-Mead', 
                              options={'maxfev': self.budget - self.evaluations})
            self.evaluations += result.nfev