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
        x0 = np.random.uniform(lb, ub, self.dim)
        
        # Dynamic perturbation scaling: Adjusts intensity based on convergence rate
        def dynamic_perturbation(x, rate):
            scale = 0.01 if rate < 0.5 else 0.005
            return x + np.random.normal(0, scale, self.dim)

        # Callback to count function evaluations and adjust perturbation
        def callback(xk):
            self.evals += 1
            nonlocal x0
            x0 = dynamic_perturbation(xk, self.evals / self.budget)

        # Define a bounded Nelder-Mead optimization process
        def bounded_nelder_mead(func, x0, bounds, maxiter):
            res = minimize(
                func, x0, method='Nelder-Mead', callback=callback,
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            # Ensure the solution is within bounds
            x_opt = np.clip(res.x, bounds.lb, bounds.ub)
            return x_opt, res.fun

        # Iteratively refine bounds and optimize
        best_x, best_f = x0, float('inf')
        remaining_budget = self.budget

        while remaining_budget > 0:
            maxiter = min(remaining_budget, 100)
            x_opt, f_opt = bounded_nelder_mead(func, x0, func.bounds, maxiter)

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