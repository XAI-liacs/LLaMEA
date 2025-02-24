import numpy as np
from scipy.optimize import minimize

class ABTELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.used_budget = 0

    def __call__(self, func):
        # Initial uniform sampling within bounds
        num_initial_samples = max(1, min(self.budget // 15, 15))  # Adjusted sample size for balance
        best_solution = None
        best_value = float('inf')

        # Sample initial points uniformly within bounds
        for _ in range(num_initial_samples):
            initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            value = func(initial_guess)
            self.used_budget += 1
            if value < best_value:
                best_value = value
                best_solution = initial_guess

        # Local optimization using BFGS or Nelder-Mead
        def local_optimize(x0):
            nonlocal best_solution, best_value
            result = minimize(func, x0, method='BFGS', bounds=[func.bounds.lb, func.bounds.ub])
            self.used_budget += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        # Iteratively refine solution
        while self.used_budget < self.budget:
            local_optimize(best_solution)

            # Adaptive bound tuning
            new_lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (func.bounds.ub - func.bounds.lb))
            new_ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (func.bounds.ub - func.bounds.lb))

            # Constrain the search space further (if budget allows)
            if self.used_budget < self.budget - 1:
                result = minimize(func, best_solution, method='Nelder-Mead', bounds=[new_lb, new_ub])
                self.used_budget += result.nfev
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x

        return best_solution