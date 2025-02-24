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
        gradient_factor = 0.02  # Adjusted gradient perturbation factor

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                perturbed_point = point + gradient_factor * np.random.randn(self.dim)  # Adjusted gradient perturbation
                result = minimize(func, perturbed_point, method='Nelder-Mead', options={
                    'maxfev': min(50, self.budget - self.evaluations),
                    'fatol': 1e-8, 'xatol': 1e-8})  # Further reduced tolerance
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                    adaptive_factor *= 0.9  # Decrease adaptive factor more aggressively on improvement
                self.evaluations += result.nfev

            # Adaptive reset with refined stochastic search space
            noise = 0.05 * (bounds[:, 1] - bounds[:, 0]) * np.random.randn(self.dim)  # Added stochastic noise
            refined_bounds = np.array([
                np.clip(best_point - adaptive_factor + noise, bounds[:, 0], bounds[:, 1]),
                np.clip(best_point + adaptive_factor + noise, bounds[:, 0], bounds[:, 1])
            ]).T
            initial_points = np.random.uniform(refined_bounds[:, 0], refined_bounds[:, 1], (self.dim + 1, self.dim))

        return best_point