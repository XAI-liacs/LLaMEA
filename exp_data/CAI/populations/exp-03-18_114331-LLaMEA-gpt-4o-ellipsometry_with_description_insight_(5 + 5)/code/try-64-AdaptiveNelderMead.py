import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial uniform sampling within bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_samples = self.budget // 3  # Adjusted sampling size
        points = np.random.uniform(lb, ub, (initial_samples, self.dim))
        
        # Track best found solution
        best_point = None
        best_value = float('inf')
        
        evaluations = 0
        
        for p in points:
            if evaluations >= self.budget:
                break

            # Nelder-Mead optimization from each start point
            result = minimize(func, p, method='Nelder-Mead',
                              options={'maxfev': self.budget - evaluations,
                                       'xatol': 1e-8, 'fatol': 1e-8})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_point = result.x

            # Adjust bounds around new best found solution
            adjustment_factor = max(0.01, 0.05 * (evaluations / self.budget))  # Dynamic adjustment factor
            lb = np.maximum(best_point - (ub - lb) * adjustment_factor, func.bounds.lb)
            ub = np.minimum(best_point + (ub - lb) * adjustment_factor, func.bounds.ub)

        return best_point