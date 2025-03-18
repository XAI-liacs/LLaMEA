import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initial uniform sampling within bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        # Increased sampling density for better coverage
        points = np.random.uniform(lb, ub, (self.budget//5, self.dim))
        
        # Track best found solution
        best_point = None
        best_value = float('inf')
        
        evaluations = 0
        
        # Reduced initial exploration to free budget for deeper exploitation
        exploration_phase = self.budget // 10
        exploration_points = np.random.uniform(lb, ub, (exploration_phase, self.dim))
        
        for p in exploration_points:
            result = minimize(func, p, method='Nelder-Mead',
                              options={'maxfev': exploration_phase,
                                       'xatol': 1e-8, 'fatol': 1e-8})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_point = result.x

        while evaluations < self.budget:
            result = minimize(func, best_point, method='Nelder-Mead',
                              options={'maxfev': self.budget - evaluations,
                                       'xatol': 1e-8, 'fatol': 1e-8})
            evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_point = result.x

            # Adjust bounds around new best found solution with dynamic step size
            adjustment_factor = 0.02 + 0.01 * (evaluations / self.budget)
            lb = np.maximum(best_point - (ub - lb) * adjustment_factor, func.bounds.lb)
            ub = np.minimum(best_point + (ub - lb) * adjustment_factor, func.bounds.ub)

        return best_point