import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class EnhancedHybridSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Latin Hypercube Sampling for initial exploration
        num_initial_samples = min(5, self.budget // 2)
        sampler = qmc.LatinHypercube(d=self.dim)
        sample_points = sampler.random(n=num_initial_samples)
        scaled_points = qmc.scale(sample_points, bounds[0], bounds[1])

        for point in scaled_points:
            value = func(point)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = point

        # Local optimization using BFGS with dynamic bounds adjustment
        remaining_budget = self.budget - evaluations
        if remaining_budget > 0:
            def callback(xk):
                nonlocal evaluations, best_solution, best_value
                value = func(xk)
                evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = xk
                return evaluations >= self.budget

            options = {'maxiter': remaining_budget, 'gtol': 1e-8}
            adjusted_bounds = list(zip(
                np.maximum(bounds[0], best_solution - 0.1 * (bounds[1] - bounds[0])),
                np.minimum(bounds[1], best_solution + 0.1 * (bounds[1] - bounds[0]))
            ))
            result = minimize(func, best_solution, method='L-BFGS-B', callback=callback, options=options, bounds=adjusted_bounds)
            if result.fun < best_value:
                best_solution = result.x

        return best_solution