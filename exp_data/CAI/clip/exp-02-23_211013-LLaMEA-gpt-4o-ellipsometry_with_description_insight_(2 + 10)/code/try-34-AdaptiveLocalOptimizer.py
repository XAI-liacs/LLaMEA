import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol, LatinHypercube

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
        
        # Combine Sobol and Latin Hypercube Sampling for initial points
        sobol_engine = Sobol(d=self.dim, scramble=True)
        lhs_engine = LatinHypercube(d=self.dim)
        
        num_sobol_points = 8
        num_lhs_points = 8
        initial_points_sobol = sobol_engine.random_base2(m=int(np.log2(num_sobol_points))) * (ub - lb) + lb
        initial_points_lhs = lhs_engine.random(n=num_lhs_points) * (ub - lb) + lb
        initial_points = np.vstack((initial_points_sobol, initial_points_lhs))
        
        for point in initial_points:
            if current_budget >= self.budget:
                break

            res = minimize(func, point, method='L-BFGS-B', bounds=zip(lb, ub))
            current_budget += res.nfev

            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun
                
        return best_solution