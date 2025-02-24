import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        starts = self.budget // (self.dim * 10)  # Estimate initial starts
        
        for _ in range(starts):
            # Generate a random initial point within the bounds
            x0 = np.random.uniform(lb, ub, self.dim)
            
            # Define a callback to stop when budget is exhausted
            def callback(xk):
                nonlocal evaluations
                evaluations += 1
                return evaluations >= self.budget
            
            # Run a local optimizer from the initial point
            res = minimize(func, x0, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)], callback=callback)
            
            # Check if we have exhausted the budget
            if evaluations >= self.budget:
                break
            
            # Update the best solution found
            if res.fun < best_value:
                best_solution = res.x
                best_value = res.fun

            # Adjust the bounds adaptively based on the current solution
            lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
            ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))

        return best_solution