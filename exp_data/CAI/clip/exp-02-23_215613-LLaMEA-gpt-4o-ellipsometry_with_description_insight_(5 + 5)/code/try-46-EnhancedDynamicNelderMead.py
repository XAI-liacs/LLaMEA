import numpy as np
from scipy.optimize import minimize

class EnhancedDynamicNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
        best_point = initial_points[0]
        best_value = float('inf')
        adaptive_factor = 0.1 * (bounds[:, 1] - bounds[:, 0])

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                # Integrating gradient descent for refinement
                result = minimize(func, point, method='Nelder-Mead', options={
                    'maxfev': self.budget - self.evaluations,
                    'fatol': 1e-7, 'xatol': 1e-7})
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                    adaptive_factor *= 0.9  # Adjust to encourage exploitation
                self.evaluations += result.nfev

            # Improved reset with dynamic bounds contraction
            initial_points = np.random.uniform(np.maximum(bounds[:, 0], best_point - adaptive_factor),
                                               np.minimum(bounds[:, 1], best_point + adaptive_factor),
                                               (self.dim + 1, self.dim))
            # Introduce small random perturbation
            initial_points += np.random.normal(0, 0.01, initial_points.shape)  

        return best_point