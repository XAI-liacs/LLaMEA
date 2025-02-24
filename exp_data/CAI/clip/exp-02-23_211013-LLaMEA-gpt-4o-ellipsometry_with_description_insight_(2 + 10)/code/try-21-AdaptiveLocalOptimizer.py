import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the search space
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Initialize variables
        current_budget = 0
        best_solution = None
        best_score = float('inf')
        
        # Uniformly sample initial points to cover the parameter space
        initial_points = np.random.uniform(lb, ub, (10, self.dim))
        
        for point in initial_points:
            if current_budget >= self.budget:
                break

            # Define a local optimization procedure
            res = minimize(func, point, method='L-BFGS-B', bounds=zip(lb, ub))
            current_budget += res.nfev  # Number of function evaluations
            
            # Update the best solution found
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun
            
            # Dynamically adjust bounds and constraints if needed
            if best_solution is not None:  # Check for existing best solution
                lb = np.maximum(lb, best_solution - 0.1*np.abs(best_solution))
                ub = np.minimum(ub, best_solution + 0.1*np.abs(best_solution))

        return best_solution