import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        perturbation_scale = 0.01  # Initial perturbation scale

        def callback(xk):
            self.evals += 1

        def bounded_nelder_mead(func, x0, bounds, maxiter):
            res = minimize(
                func, x0, method='Nelder-Mead', callback=callback,
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            x_opt = np.clip(res.x, bounds.lb, bounds.ub)
            return x_opt, res.fun

        best_x, best_f = None, float('inf')
        remaining_budget = self.budget

        while remaining_budget > 0:
            x0 = np.random.uniform(lb, ub, self.dim) + np.random.normal(0, perturbation_scale, self.dim)
            maxiter = min(remaining_budget, 100)
            x_opt, f_opt = bounded_nelder_mead(func, x0, func.bounds, maxiter)

            if f_opt < best_f:
                best_x, best_f = x_opt, f_opt
                x0 = best_x
                r = 0.1 * (ub - lb)
                lb, ub = np.maximum(func.bounds.lb, best_x - r), np.minimum(func.bounds.ub, best_x + r)
                perturbation_scale *= 0.9  # Reduce perturbation scale on improvement
            else:
                perturbation_scale *= 1.1  # Increase perturbation on stagnation
                if self.evals % 50 == 0:  # Random restart if stagnation persists
                    x0 = np.random.uniform(lb, ub, self.dim)

            remaining_budget -= maxiter
            if self.evals >= self.budget:
                break

        return best_x