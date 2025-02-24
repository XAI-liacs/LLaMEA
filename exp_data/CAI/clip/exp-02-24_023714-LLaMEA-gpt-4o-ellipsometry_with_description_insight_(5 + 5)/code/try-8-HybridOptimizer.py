import numpy as np
from scipy.optimize import minimize, Bounds

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def _update_bounds(self, best_x, scale=0.1):
        lb = np.maximum(self.func_bounds.lb, best_x - scale*(self.func_bounds.ub - self.func_bounds.lb))
        ub = np.minimum(self.func_bounds.ub, best_x + scale*(self.func_bounds.ub - self.func_bounds.lb))
        return Bounds(lb, ub)

    def __call__(self, func):
        self.func_bounds = func.bounds
        
        while self.evals < self.budget:
            initial_guess = np.random.uniform(self.func_bounds.lb, self.func_bounds.ub)
            res_nm = minimize(func, initial_guess, method='Nelder-Mead', options={'maxiter': self.budget//4, 'disp': False})
            self.evals += res_nm.nfev
            
            if self.evals >= self.budget:
                return res_nm.x, res_nm.fun

            adaptive_bounds = self._update_bounds(res_nm.x)
            res_bfgs = minimize(func, res_nm.x, method='L-BFGS-B', bounds=adaptive_bounds, options={'maxfun': self.budget - self.evals, 'disp': False})
            self.evals += res_bfgs.nfev

            if self.evals >= self.budget:
                return res_bfgs.x, res_bfgs.fun

            if np.random.rand() < 0.1:
                continue  # Random restart with probability 0.1
        
        return res_bfgs.x, res_bfgs.fun