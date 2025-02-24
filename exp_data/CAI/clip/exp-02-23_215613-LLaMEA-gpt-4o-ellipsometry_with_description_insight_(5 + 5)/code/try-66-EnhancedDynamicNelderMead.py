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
        adaptive_factor = 0.1 * (bounds[:, 1] - bounds[:, 0])  # Adaptive factor for reset
        gradient_factor = 0.01  # New gradient perturbation factor

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                perturbed_point = point + gradient_factor * np.random.randn(self.dim)  # Gradient perturbation
                result = minimize(func, perturbed_point, method='Nelder-Mead', options={
                    'maxfev': self.budget - self.evaluations,
                    'fatol': 1e-7, 'xatol': 1e-7})  # Reduced tolerance
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                    adaptive_factor *= 0.95  # Decrease adaptive factor on improvement
                self.evaluations += result.nfev

            # Adaptive reset with refined search space based on best point
            initial_points = np.random.uniform(np.maximum(bounds[:, 0], best_point - adaptive_factor),
                                               np.minimum(bounds[:, 1], best_point + adaptive_factor),
                                               (self.dim + 1, self.dim))
            gradient_factor *= 0.9  # Refine gradient perturbation on each iteration

        return best_point