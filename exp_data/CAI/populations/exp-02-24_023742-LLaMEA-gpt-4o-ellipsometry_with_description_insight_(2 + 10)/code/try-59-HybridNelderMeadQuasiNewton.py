import numpy as np
from scipy.optimize import minimize

class HybridNelderMeadQuasiNewton:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Initialize bounds and starting point
        lb, ub = func.bounds.lb, func.bounds.ub
        x0 = np.random.uniform(lb, ub, self.dim)
        
        # Improved initialization with Sobol sequence for diverse starting points
        sobol_seq = np.random.rand(self.dim)*0.5*(ub-lb) + 0.5*(lb+ub)
        x0 = np.mean([x0, sobol_seq], axis=0)

        # Callback to count function evaluations
        def callback(xk):
            self.evals += 1

        # Define a bounded Nelder-Mead optimization process
        def bounded_nelder_mead(func, x0, bounds, maxiter):
            res = minimize(
                func, x0, method='Nelder-Mead', callback=callback,
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            # Ensure the solution is within bounds
            x_opt = np.clip(res.x, bounds.lb, bounds.ub)
            return x_opt, res.fun

        # Define a Quasi-Newton optimization process
        def bounded_quasi_newton(func, x0, bounds, maxiter):
            res = minimize(
                func, x0, method='L-BFGS-B', bounds=[(l, u) for l, u in zip(bounds.lb, bounds.ub)],
                options={'maxiter': maxiter}
            )
            return res.x, res.fun

        # Iteratively refine bounds and optimize with hybrid approach
        best_x, best_f = x0, float('inf')
        remaining_budget = self.budget

        while remaining_budget > 0:
            if self.evals > 0 and self.evals % 50 == 0:  # Hybrid switch every 50 evaluations
                x_opt, f_opt = bounded_quasi_newton(func, x0, func.bounds, remaining_budget)
            else:
                maxiter = min(remaining_budget, 100)
                x_opt, f_opt = bounded_nelder_mead(func, x0, func.bounds, maxiter)
            
            if f_opt < best_f:
                best_x, best_f = x_opt, f_opt
                # Refine the search space around the best found solution
                x0 = best_x
                r = 0.05 * (ub - lb)  # Reduced step size for finer refinement
                lb, ub = np.maximum(func.bounds.lb, best_x - r), np.minimum(func.bounds.ub, best_x + r)
            
            remaining_budget -= 50 if self.evals > 0 and self.evals % 50 == 0 else maxiter
            if self.evals >= self.budget:
                break
        
        return best_x