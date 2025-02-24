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
                result = minimize(func, point, method='Nelder-Mead', options={
                    'maxfev': self.budget - self.evaluations,
                    'fatol': 1e-8, 'xatol': 1e-8})  # Tighter tolerances
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                    adaptive_factor *= 0.85
                self.evaluations += result.nfev

            gradient_step = 0.03 * (bounds[:, 1] - bounds[:, 0])
            perturbation_factor = np.random.normal(0, gradient_step, (self.dim + 1, self.dim))
            gradient_points = best_point + perturbation_factor
            for g_point in gradient_points:
                if self.evaluations >= self.budget:
                    break
                result = minimize(func, g_point, method='Nelder-Mead', options={
                    'maxfev': self.budget - self.evaluations,
                    'fatol': 1e-8, 'xatol': 1e-8})  # Tighter tolerances
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                self.evaluations += result.nfev

            restart_points = np.random.uniform(
                np.maximum(bounds[:, 0], best_point - 1.1 * adaptive_factor),  # Reduced range
                np.minimum(bounds[:, 1], best_point + 1.1 * adaptive_factor),  # Reduced range
                (self.dim + 1, self.dim)) + np.random.normal(0, adaptive_factor, (self.dim + 1, self.dim))
            initial_points = np.clip(restart_points, bounds[:, 0], bounds[:, 1])

        return best_point