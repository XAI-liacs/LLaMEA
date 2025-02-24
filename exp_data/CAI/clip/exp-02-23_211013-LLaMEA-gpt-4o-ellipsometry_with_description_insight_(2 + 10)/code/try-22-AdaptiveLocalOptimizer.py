import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from skopt import gp_minimize

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
        
        sobol_engine = Sobol(d=self.dim, scramble=True)
        initial_points = sobol_engine.random_base2(m=3) * (ub - lb) + lb
        
        points_evaluated = []
        scores_evaluated = []
        
        for point in initial_points:
            if current_budget >= self.budget:
                break

            res = minimize(func, point, method='L-BFGS-B', bounds=zip(lb, ub))
            current_budget += res.nfev
            
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun
                
            points_evaluated.append(res.x)
            scores_evaluated.append(res.fun)
        
        if current_budget < self.budget:
            gp_res = gp_minimize(func, [(l, u) for l, u in zip(lb, ub)], n_calls=self.budget-current_budget, 
                                 x0=points_evaluated, y0=scores_evaluated)
            if gp_res.fun < best_score:
                best_solution = gp_res.x
                best_score = gp_res.fun

        return best_solution