import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Initialize bounds and starting point
        lb, ub = func.bounds.lb, func.bounds.ub
        x0 = np.random.uniform(lb, ub, self.dim) + np.random.normal(0, 0.05, self.dim)  # Adjusted perturbation

        # Callback to count function evaluations
        def callback(xk):
            self.evals += 1

        # Define a bounded Nelder-Mead optimization process
        def bounded_nelder_mead(func, x0, bounds, maxiter, adaptive_scale):  # Added adaptive_scale parameter
            options = {
                'maxiter': maxiter, 
                'xatol': 1e-8, 
                'fatol': 1e-8,
                'initial_simplex': np.clip(np.outer(np.ones(self.dim + 1), x0) + 
                                           adaptive_scale * np.random.normal(0, 1, (self.dim + 1, self.dim)), 
                                           bounds.lb, bounds.ub)  # Initial simplex scaling
            }
            res = minimize(
                func, x0, method='Nelder-Mead', callback=callback,
                options=options
            )
            # Ensure the solution is within bounds
            x_opt = np.clip(res.x, bounds.lb, bounds.ub)
            return x_opt, res.fun

        # Iteratively refine bounds and optimize
        best_x, best_f = x0, float('inf')
        remaining_budget = self.budget

        while remaining_budget > 0:
            maxiter = min(remaining_budget, 100)
            adaptive_scale = 0.1 * (remaining_budget / self.budget)  # Dynamic scaling of initial simplex
            x_opt, f_opt = bounded_nelder_mead(func, x0, func.bounds, maxiter, adaptive_scale)
            
            if f_opt < best_f:
                best_x, best_f = x_opt, f_opt
                # Refine the search space around the best found solution
                x0 = best_x
                r = 0.1 * (ub - lb)
                lb, ub = np.maximum(func.bounds.lb, best_x - r), np.minimum(func.bounds.ub, best_x + r)
            
            remaining_budget -= maxiter
            if self.evals >= self.budget:
                break
        
        return best_x