import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        points = np.random.uniform(lb, ub, (self.budget//5, self.dim))
        
        best_point = None
        best_value = float('inf')
        
        evaluations = 0
        restart_threshold = 1e-7  # Refined threshold for dynamic restart

        for p in points:
            if evaluations >= self.budget:
                break

            result = minimize(func, p, method='Nelder-Mead',
                              options={'maxfev': self.budget - evaluations,
                                       'xatol': 1e-8, 'fatol': 1e-8})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_point = result.x

            if abs(result.fun - best_value) < restart_threshold:
                continue  # Early exit if improvement is minimal

            adjustment_factor = max(0.01, min(0.05, 0.02 + 0.03 * (self.budget - evaluations) / self.budget))  # Adjusted dynamic factor
            lb = np.maximum(best_point - (ub - lb) * adjustment_factor, func.bounds.lb)
            ub = np.minimum(best_point + (ub - lb) * adjustment_factor, func.bounds.ub)

        return best_point