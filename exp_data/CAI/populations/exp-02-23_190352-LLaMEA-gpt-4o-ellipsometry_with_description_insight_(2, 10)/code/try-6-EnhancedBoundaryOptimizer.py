import numpy as np
from scipy.optimize import minimize

class EnhancedBoundaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds_lower = func.bounds.lb
        bounds_upper = func.bounds.ub

        def bounded_func(x):
            x_clipped = np.clip(x, bounds_lower, bounds_upper)
            return func(x_clipped)

        def dynamic_bounds_shrink(bounds_lower, bounds_upper, current_best):
            center = current_best
            span = (bounds_upper - bounds_lower) * 0.1
            new_lower = np.maximum(bounds_lower, center - span)
            new_upper = np.minimum(bounds_upper, center + span)
            return new_lower, new_upper

        num_restarts = 5
        evals_per_restart = self.budget // num_restarts
        best_solution = None
        best_value = np.inf

        for _ in range(num_restarts):
            x0 = np.random.uniform(bounds_lower, bounds_upper, self.dim)
            options = {'maxiter': evals_per_restart, 'adaptive': True}
            result = minimize(bounded_func, x0, method='Nelder-Mead', options=options)

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            bounds_lower, bounds_upper = dynamic_bounds_shrink(bounds_lower, bounds_upper, result.x)

        return best_solution

# Example usage:
# optimizer = EnhancedBoundaryOptimizer(budget=100, dim=2)
# best_solution = optimizer(your_black_box_function)