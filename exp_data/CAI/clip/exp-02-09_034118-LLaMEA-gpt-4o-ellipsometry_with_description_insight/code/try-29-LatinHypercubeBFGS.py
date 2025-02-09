import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

class LatinHypercubeBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Initial sampling using Latin Hypercube for diverse exploration
        num_initial_samples = min(10, self.budget // 2)
        lhs = LatinHypercube(d=self.dim)
        initial_samples = lhs.random(n=num_initial_samples)
        initial_samples = bounds[0] + (bounds[1] - bounds[0]) * initial_samples

        for x0 in initial_samples:
            value = func(x0)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0

        # Local optimization using BFGS
        remaining_budget = self.budget - evaluations
        if remaining_budget > 0:
            def callback(xk):
                nonlocal evaluations, best_solution, best_value
                if evaluations >= self.budget:
                    return True
                value = func(xk)
                evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = xk

            options = {'maxiter': remaining_budget, 'gtol': 1e-8}
            result = minimize(func, best_solution, method='BFGS', callback=callback, options=options, bounds=bounds.T)
            if result.fun < best_value:
                best_solution = result.x

        return best_solution