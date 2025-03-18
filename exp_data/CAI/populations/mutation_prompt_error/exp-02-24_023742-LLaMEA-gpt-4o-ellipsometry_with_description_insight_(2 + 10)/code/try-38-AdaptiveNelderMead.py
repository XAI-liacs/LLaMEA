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
        
        # Improved initialization with Halton sequence for diverse starting points
        def halton_sequence(size, dim):
            def van_der_corput(n, base=2):
                vdc, denom = 0, 1
                while n:
                    n, remainder = divmod(n, base)
                    denom *= base
                    vdc += remainder / float(denom)
                return vdc

            return np.array([[van_der_corput(i, base=j + 2) for j in range(dim)] for i in range(size)])

        halton_seq = halton_sequence(1, self.dim)[0] * (ub - lb) + lb
        x0 = np.mean([x0, halton_seq], axis=0)

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
                r = 0.05 * (ub - lb)  # Reduced step size for finer refinement
                lb, ub = np.maximum(func.bounds.lb, best_x - r), np.minimum(func.bounds.ub, best_x + r)
            
            remaining_budget -= maxiter
            if self.evals >= self.budget:
                break
        
        return best_x