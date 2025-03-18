import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))
        
        num_initial_samples = min(25, self.budget // 8)  # Changed from 20 to 25 and budget division factor
        
        initial_samples = np.random.uniform(lb, ub, size=(num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for i, sample in enumerate(initial_samples):
            if self.evaluations >= self.budget:
                break
            
            # Use both L-BFGS-B and Nelder-Mead for local optimization
            res_lbfgsb = minimize(func, sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            if self.evaluations + res_lbfgsb.nfev < self.budget:
                res_nm = minimize(func, res_lbfgsb.x, method='Nelder-Mead', options={'maxfev': self.budget - self.evaluations - res_lbfgsb.nfev})
                self.evaluations += res_lbfgsb.nfev + res_nm.nfev
            else:
                res_nm = res_lbfgsb
                self.evaluations += res_lbfgsb.nfev
            
            if res_nm.fun < best_value:
                best_value = res_nm.fun
                best_solution = res_nm.x
            
            radius = 0.05 * (ub - lb)
            lb = np.maximum(lb, best_solution - radius)
            ub = np.minimum(ub, best_solution + radius)
            bounds = list(zip(lb, ub))
        
        return best_solution