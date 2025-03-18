import numpy as np
from scipy.optimize import minimize

class ImprovedAdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = np.inf
        evaluations = 0

        # Initial uniform random sampling
        num_initial_samples = min(5, self.budget // 3)
        for _ in range(num_initial_samples):
            x0 = np.random.uniform(bounds[0], bounds[1], self.dim)
            value = func(x0)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0

        # Choose optimization method based on initial findings
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

            # Dynamic choice of optimization method
            method = 'BFGS' if self.dim > 1 else 'Nelder-Mead'
            options = {'maxiter': remaining_budget, 'gtol': 1e-5, 'disp': False} if method == 'BFGS' else {'maxiter': remaining_budget, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            
            result = minimize(func, best_solution, method=method, callback=callback, options=options, bounds=bounds.T)
            if result.fun < best_value:
                best_solution = result.x

        return best_solution