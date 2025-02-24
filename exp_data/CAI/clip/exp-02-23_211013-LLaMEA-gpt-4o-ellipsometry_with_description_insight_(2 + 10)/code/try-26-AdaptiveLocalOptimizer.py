import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

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
        sobol_engine = Sobol(d=self.dim, scramble=True)
        initial_points = sobol_engine.random_base2(m=3) * (ub - lb) + lb  # Reduced initial samples
        
        for point in initial_points:
            if current_budget >= self.budget:
                break

            # Define a local optimization procedure
            dynamic_bounds = [(max(l, x - 0.1 * (u - l)), min(u, x + 0.1 * (u - l))) for x, l, u in zip(point, lb, ub)]
            res = minimize(func, point, method='L-BFGS-B', bounds=dynamic_bounds)  # Use dynamic bounds
            current_budget += res.nfev
            
            # Update the best solution found
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun
            
        return best_solution