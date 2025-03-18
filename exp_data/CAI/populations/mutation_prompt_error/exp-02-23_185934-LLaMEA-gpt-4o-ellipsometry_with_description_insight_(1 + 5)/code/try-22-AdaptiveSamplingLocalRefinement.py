import numpy as np
from scipy.optimize import minimize

class AdaptiveSamplingLocalRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.grid_samples = min(10, budget // 2)

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Adaptive grid sampling
        grid_points = np.linspace(lb, ub, self.grid_samples)
        initial_sample_count = self.grid_samples
        for point in np.nditer(np.meshgrid(*[grid_points[:, i] for i in range(self.dim)])):
            x0 = np.array(point)
            value = func(x0)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0
                # Early local refinement on promising points
                if self.evaluations < self.budget // 2:
                    res = minimize(func, x0, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                                   options={'maxiter': self.budget // 10, 'gtol': 1e-5})  # Early refinement with threshold
                    if res.fun < best_value:
                        best_value = res.fun
                        best_solution = res.x
                        
            # Dynamically adjust grid points based on initial evaluations
            if self.evaluations == (initial_sample_count // 2):
                self.grid_samples = min(self.grid_samples + 5, self.budget // 2)
                grid_points = np.linspace(lb, ub, self.grid_samples)

            if self.evaluations >= self.budget:
                return best_solution

        # Local optimization using BFGS with trust-region strategy
        remaining_budget = self.budget - self.evaluations
        if remaining_budget > 0:
            res = minimize(func, best_solution, method='trust-constr', bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': remaining_budget})
            if res.fun < best_value:
                best_solution = res.x

        return best_solution