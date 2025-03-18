import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Initial uniform random sampling
        num_initial_samples = min(20, self.budget // 3)  # Adjusted number of initial samples
        for _ in range(num_initial_samples):
            x0 = np.random.uniform(bounds[0], bounds[1], self.dim)
            value = func(x0)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0

        # Local optimization using Nelder-Mead
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
                return np.abs(value - best_value) < 1e-7  # Refined stopping criterion

            options = {'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-8}
            result = minimize(func, best_solution, method='Nelder-Mead', callback=callback, options=options, bounds=bounds.T)
            if result.fun < best_value:
                best_solution = result.x

        return best_solution