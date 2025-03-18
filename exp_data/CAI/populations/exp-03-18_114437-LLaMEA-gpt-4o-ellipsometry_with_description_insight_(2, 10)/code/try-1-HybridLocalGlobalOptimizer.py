import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Uniform sampling for initial solutions
        num_initial_samples = min(10, self.budget // 2)
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        sample_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Evaluate initial samples
        for point in sample_points:
            if evaluations >= self.budget:
                break
            value = func(point)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = point

        # Local optimization with budget constraint using BFGS
        remaining_budget = self.budget - evaluations
        if remaining_budget > 0:
            def limited_func(x):
                nonlocal evaluations
                if evaluations >= self.budget:
                    return best_value  # Return worst case if over budget
                evaluations += 1
                return func(x)

            options = {'maxiter': remaining_budget, 'disp': False}
            result = minimize(limited_func, best_solution, method='L-BFGS-B', bounds=bounds, options=options)

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution