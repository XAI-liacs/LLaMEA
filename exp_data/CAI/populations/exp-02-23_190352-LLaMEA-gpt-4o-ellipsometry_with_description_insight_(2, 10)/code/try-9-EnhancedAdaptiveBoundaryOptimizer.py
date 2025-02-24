import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveBoundaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds_lower = np.array(func.bounds.lb, dtype=float)
        bounds_upper = np.array(func.bounds.ub, dtype=float)
        best_solution = None
        best_value = float('inf')
        evaluations_used = 0
        
        while evaluations_used < self.budget:
            # Create a uniform random sample for the initial guess within bounds
            x0 = np.random.uniform(bounds_lower, bounds_upper, self.dim)
            
            def bounded_func(x):
                x_clipped = np.clip(x, bounds_lower, bounds_upper)
                return func(x_clipped)

            options = {
                'maxiter': min(self.budget - evaluations_used, 100),  # Limit iterations per Nelder-Mead run
                'adaptive': True
            }

            result = minimize(bounded_func, x0, method='Nelder-Mead', options=options)
            evaluations_used += result.nfev

            if result.success and result.fun < best_value:
                best_solution = result.x
                best_value = result.fun
                # Shrink bounds towards the current best solution
                bounds_lower = np.maximum(bounds_lower, best_solution - (bounds_upper - bounds_lower) * 0.1)
                bounds_upper = np.minimum(bounds_upper, best_solution + (bounds_upper - bounds_lower) * 0.1)

        return best_solution

# Example usage:
# optimizer = EnhancedAdaptiveBoundaryOptimizer(budget=100, dim=2)
# best_solution = optimizer(your_black_box_function)