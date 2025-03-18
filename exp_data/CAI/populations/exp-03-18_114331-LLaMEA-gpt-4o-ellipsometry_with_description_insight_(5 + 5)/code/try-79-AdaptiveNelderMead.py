import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        points = np.random.uniform(lb, ub, (self.budget//4, self.dim))  # Changed sampling density

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

            if evaluations < self.budget * 0.8:  # Introduced local exploitation condition
                local_result = minimize(
                    func, best_point, method='L-BFGS-B', 
                    bounds=list(zip(lb, ub)), options={'maxfun': 10}
                )
                evaluations += local_result.nfev
                if local_result.fun < best_value:
                    best_value = local_result.fun
                    best_point = local_result.x

            adjustment_factor = 0.05 + 0.005 * (self.budget - evaluations) / self.budget  # Adjusted factor
            lb = np.maximum(best_point - (ub - lb) * adjustment_factor, func.bounds.lb)
            ub = np.minimum(best_point + (ub - lb) * adjustment_factor, func.bounds.ub)

        return best_point