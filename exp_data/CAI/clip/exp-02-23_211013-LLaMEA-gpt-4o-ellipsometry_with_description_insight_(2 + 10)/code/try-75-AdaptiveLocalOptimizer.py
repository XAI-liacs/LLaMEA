import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        current_budget = 0
        best_solution = None
        best_score = float('inf')
        
        # Use Sobol sequence for initial sampling
        sobol_engine = Sobol(d=self.dim, scramble=True)
        initial_points = sobol_engine.random_base2(m=4) * (ub - lb) + lb
        
        for point in initial_points:
            if current_budget >= self.budget:
                break

            # Alternate between L-BFGS-B and Nelder-Mead
            if current_budget % 2 == 0:
                method = 'L-BFGS-B'
            else:
                method = 'Nelder-Mead'

            # Local optimization using selected method
            res = minimize(func, point, method=method, bounds=zip(lb, ub))
            current_budget += res.nfev
            
            # Update the best solution
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun
            
            # Adaptive adjustment of bounds based on budget usage
            if current_budget < 0.5 * self.budget:
                lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
                ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))

        return best_solution