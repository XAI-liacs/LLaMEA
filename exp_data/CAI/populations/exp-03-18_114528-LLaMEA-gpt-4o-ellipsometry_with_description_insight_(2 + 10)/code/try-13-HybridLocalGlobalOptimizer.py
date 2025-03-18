import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initial uniform sampling to get diverse starting points
        num_initial_samples = min(10, self.budget // 3)
        initial_points = np.random.uniform(lb, ub, (num_initial_samples, self.dim))

        best_solution = None
        best_value = float('inf')
        evaluations = 0
        remaining_budget = self.budget

        for point in initial_points:
            # Dynamic allocation of evaluations for better budget usage
            local_budget = max(2, remaining_budget // num_initial_samples)

            # Local optimization using Nelder-Mead
            res = minimize(func, point, method='Nelder-Mead',
                           bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': local_budget})
            evaluations += res.nfev
            remaining_budget -= res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Adaptive restart mechanism to allow re-evaluation with new points
            if evaluations < self.budget:
                additional_points = np.random.uniform(lb, ub, (1, self.dim))
                initial_points = np.append(initial_points, additional_points, axis=0)

            if evaluations >= self.budget:
                break

        return best_solution