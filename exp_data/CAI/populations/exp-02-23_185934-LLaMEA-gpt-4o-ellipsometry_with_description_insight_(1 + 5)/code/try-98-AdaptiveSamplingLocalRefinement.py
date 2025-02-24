import numpy as np
from scipy.optimize import minimize

class AdaptiveSamplingLocalRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.grid_samples = min(10, budget // 3)  # Adjusted initial sample size

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Dynamic adaptive grid sampling
        grid_points = np.random.uniform(lb, ub, (self.grid_samples, self.dim))  # Randomized initial sampling
        for i in range(self.grid_samples):
            x0 = grid_points[i]
            value = func(x0)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0
                # Hybrid local refinement using 'trust-constr' earlier when more budget remains
                if self.evaluations < self.budget // 2:
                    res = minimize(func, x0, method='trust-constr', bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                                   options={'maxiter': self.budget // 15, 'gtol': 1e-6})
                    if res.fun < best_value:
                        best_value = res.fun
                        best_solution = res.x

            if self.evaluations >= self.budget:
                return best_solution

        # Adjusting grid sample size based on remaining budget
        self.grid_samples = max(5, self.budget // 20)  # New line modifying grid sample size strategy

        # Local optimization using hybrid strategy
        remaining_budget = self.budget - self.evaluations
        if remaining_budget > 0:
            res = minimize(func, best_solution, method='trust-constr', bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': remaining_budget})
            if res.fun < best_value:
                best_solution = res.x

        return best_solution