import numpy as np
from scipy.optimize import minimize, approx_fprime

class HybridNelderMeadGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Initialize bounds and starting point
        lb, ub = func.bounds.lb, func.bounds.ub
        x0 = np.random.uniform(lb, ub, self.dim) + np.random.normal(0, 0.01, self.dim)  # Perturbation added here
        
        # Callback to count function evaluations
        def callback(xk):
            self.evals += 1
        
        # Gradient approximation using finite differences
        def gradient_approx(x, f, epsilon=1e-8):
            return approx_fprime(x, f, epsilon)
        
        # Define a bounded optimization process
        def hybrid_optimization(func, x0, bounds, maxiter):
            # Use Nelder-Mead initially
            res = minimize(
                func, x0, method='Nelder-Mead', callback=callback,
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            x_opt = np.clip(res.x, bounds.lb, bounds.ub)
            
            # If budget allows and not optimal, switch to gradient approximation
            if self.evals < self.budget and res.success:
                gradient = gradient_approx(x_opt, func)
                x_opt = x_opt - 0.01 * gradient  # Small step in the direction of negative gradient
                x_opt = np.clip(x_opt, bounds.lb, bounds.ub)
                self.evals += 1  # Account for function evaluation in gradient approximation
            
            return x_opt, func(x_opt)

        # Iteratively refine bounds and optimize
        best_x, best_f = x0, float('inf')
        remaining_budget = self.budget

        while remaining_budget > 0:
            maxiter = min(remaining_budget, 100)
            x_opt, f_opt = hybrid_optimization(func, x0, func.bounds, maxiter)
            
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