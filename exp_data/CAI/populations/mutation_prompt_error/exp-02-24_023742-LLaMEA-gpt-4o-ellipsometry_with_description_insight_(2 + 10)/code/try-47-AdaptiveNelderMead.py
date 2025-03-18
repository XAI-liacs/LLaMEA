import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x0 = np.random.uniform(lb, ub, self.dim) + np.random.normal(0, 0.01, self.dim)  
        
        def callback(xk):
            self.evals += 1

        def bounded_nelder_mead(func, x0, bounds, maxiter):
            res = minimize(
                func, x0, method='Nelder-Mead', callback=callback,
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            x_opt = np.clip(res.x, bounds.lb, bounds.ub)
            return x_opt, res.fun

        best_x, best_f = x0, float('inf')
        remaining_budget = self.budget
        perturbation_scale = 0.01

        while remaining_budget > 0:
            maxiter = min(remaining_budget, 100)
            x_opt, f_opt = bounded_nelder_mead(func, x0, func.bounds, maxiter)
            
            if f_opt < best_f:
                best_x, best_f = x_opt, f_opt
                perturbation_scale = max(1e-4, perturbation_scale * 0.5)
                x0 = best_x + np.random.normal(0, perturbation_scale, self.dim)
                r = 0.1 * (ub - lb)
                lb, ub = np.maximum(func.bounds.lb, best_x - r), np.minimum(func.bounds.ub, best_x + r)
            else:
                perturbation_scale = min(0.1, perturbation_scale * 2)
                x0 = np.random.uniform(lb, ub, self.dim) + np.random.normal(0, perturbation_scale, self.dim)
            
            remaining_budget -= maxiter
            if self.evals >= self.budget:
                break
        
        return best_x