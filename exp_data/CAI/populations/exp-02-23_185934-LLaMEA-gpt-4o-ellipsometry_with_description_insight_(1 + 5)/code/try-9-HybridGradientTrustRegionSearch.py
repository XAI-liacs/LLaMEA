import numpy as np
from scipy.optimize import minimize

class HybridGradientTrustRegionSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.grid_samples = min(10, budget // 3)

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = np.zeros(self.dim)
        best_value = float('inf')
        
        # Coarse grid sampling
        grid_points = np.linspace(lb, ub, self.grid_samples)
        for point in np.nditer(np.meshgrid(*[grid_points[:, i] for i in range(self.dim)])):
            x0 = np.array(point)
            value = func(x0)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0

            if self.evaluations >= self.budget:
                return best_solution

        # Local optimization using Trust Region Reflective algorithm
        remaining_budget = self.budget - self.evaluations
        if remaining_budget > 0:
            res = minimize(func, best_solution, method='trust-constr', bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': remaining_budget})
            if res.fun < best_value:
                best_solution = res.x

        return best_solution