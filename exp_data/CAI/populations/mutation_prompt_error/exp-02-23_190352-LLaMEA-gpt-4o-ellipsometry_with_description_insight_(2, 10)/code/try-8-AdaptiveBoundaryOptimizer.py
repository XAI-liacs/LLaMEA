import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds_lower = func.bounds.lb
        bounds_upper = func.bounds.ub
        
        # Create a uniform random sample for the initial guess within bounds
        x0 = np.random.uniform(bounds_lower, bounds_upper, self.dim)
        
        def bounded_func(x):
            # Clip the parameters to keep them within bounds
            x_clipped = np.clip(x, bounds_lower, bounds_upper)
            return func(x_clipped)

        # Define the optimization options
        options = {
            'maxiter': self.budget,  # Budget for function evaluations
            'adaptive': True         # Adaptive simplex size
        }

        # Perform optimization using the Nelder-Mead algorithm
        result = minimize(bounded_func, x0, method='Nelder-Mead', options=options)

        # Use adaptive boundary shrinking based on the result
        if result.success:
            for _ in range(self.budget - result.nfev):
                # Shrink bounds towards the current best solution more frequently
                bounds_lower = np.maximum(bounds_lower, result.x - (bounds_upper - bounds_lower) * 0.05)
                bounds_upper = np.minimum(bounds_upper, result.x + (bounds_upper - bounds_lower) * 0.05)
                x0 = np.random.uniform(bounds_lower, bounds_upper, self.dim)
                result = minimize(bounded_func, x0, method='Nelder-Mead', options=options)
                if result.nfev >= self.budget:
                    break
        
        return result.x

# Example usage:
# optimizer = AdaptiveBoundaryOptimizer(budget=100, dim=2)
# best_solution = optimizer(your_black_box_function)