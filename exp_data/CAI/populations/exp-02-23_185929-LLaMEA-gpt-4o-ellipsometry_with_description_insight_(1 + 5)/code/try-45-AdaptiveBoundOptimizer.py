import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        remaining_budget = self.budget

        # Initial uniform sampling
        num_initial_samples = min(self.dim * 5, remaining_budget // 2)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        for sample in samples:
            value = func(sample)
            remaining_budget -= 1
            if value < best_value:
                best_value = value
                best_solution = sample
            if remaining_budget <= 0:
                break

        # Local optimization using Nelder-Mead with adaptive bounds
        if remaining_budget > 0:
            def constrained_func(x):
                x_clipped = np.clip(x, lb, ub)
                return func(x_clipped)

            while remaining_budget > 0:
                result = minimize(constrained_func, best_solution, method='Nelder-Mead', options={'maxfev': remaining_budget, 'xatol': 1e-9, 'fatol': 1e-9})
                remaining_budget -= result.nfev
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = np.clip(result.x, lb, ub)

                # Shrink bounds adaptively
                lb = np.maximum(lb, best_solution - (ub - lb) * 0.1)
                ub = np.minimum(ub, best_solution + (ub - lb) * 0.1)

        return best_solution