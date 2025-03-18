import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        init_points = np.random.uniform(lb, ub, size=(10, self.dim))
        
        best_solution = None
        best_value = float('inf')

        for point in init_points:
            if self.evaluations >= self.budget:
                break

            result = minimize(self.bounded_func(func, lb, ub), point, method='BFGS',
                              options={'maxiter': self.budget - self.evaluations})
            
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            dynamic_adjustment = 0.01 + 0.005*(self.evaluations/self.budget)  # Modified adjustment
            lb = np.maximum(lb, best_solution - dynamic_adjustment * (ub - lb))
            ub = np.minimum(ub, best_solution + dynamic_adjustment * (ub - lb))

        return best_solution

    def bounded_func(self, func, lb, ub):
        def func_with_bounds(x):
            x_clipped = np.clip(x, lb, ub)
            return func(x_clipped)
        return func_with_bounds