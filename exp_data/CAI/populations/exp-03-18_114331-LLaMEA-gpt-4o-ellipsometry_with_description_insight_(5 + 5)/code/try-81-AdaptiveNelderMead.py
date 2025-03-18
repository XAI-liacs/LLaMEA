import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Adjusted initial uniform sampling density based on budget usage
        sampling_density = max(5, self.budget // (10 * self.dim))
        points = np.random.uniform(lb, ub, (sampling_density, self.dim))
        
        best_point = None
        best_value = float('inf')
        
        evaluations = 0
        
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

            adjustment_factor = 0.03 + 0.005 * (self.budget - evaluations) / self.budget
            lb = np.maximum(best_point - (ub - lb) * adjustment_factor, func.bounds.lb)
            ub = np.minimum(best_point + (ub - lb) * adjustment_factor, func.bounds.ub)

        return best_point