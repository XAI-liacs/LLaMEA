import numpy as np
from scipy.optimize import minimize

class DynamicNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
        best_point = initial_points[0]
        best_value = float('inf')

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                result = minimize(func, point, method='Nelder-Mead', options={'maxfev': self.budget - self.evaluations, 'adaptive': True})
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                self.evaluations += result.nfev

            step_size = (self.budget - self.evaluations) / self.budget
            initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim)) * (1 - step_size)

        return best_point