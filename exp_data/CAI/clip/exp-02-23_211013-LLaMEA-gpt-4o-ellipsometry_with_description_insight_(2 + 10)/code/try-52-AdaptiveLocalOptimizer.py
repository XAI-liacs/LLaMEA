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
        
        # Use Sobol sequence for initial points to improve coverage
        sampler = Sobol(d=self.dim, scramble=True)
        initial_points = lb + (ub - lb) * sampler.random_base2(m=4)
        
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
                # Dynamically adjust bounds to exploit the region around the best solution
                lb, ub = np.maximum(lb, best_solution - 0.1), np.minimum(ub, best_solution + 0.1)

        return best_solution