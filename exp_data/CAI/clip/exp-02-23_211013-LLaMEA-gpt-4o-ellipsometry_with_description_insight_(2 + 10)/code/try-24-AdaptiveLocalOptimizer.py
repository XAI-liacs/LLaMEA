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
        
        # Use Sobol sequence for initial points to improve coverage
        m = min(int(np.log2(self.budget // self.dim)), 4)
        sampler = Sobol(d=self.dim, scramble=True)
        initial_points = lb + (ub - lb) * sampler.random_base2(m=m)
        
        for point in initial_points:
            if current_budget >= self.budget:
                break

            res = minimize(func, point, method='L-BFGS-B', bounds=zip(lb, ub))
            current_budget += res.nfev
            
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun

        return best_solution