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
        gradient_factor = 0.02  # Modified gradient perturbation factor

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                perturbed_point = point + gradient_factor * np.random.normal(size=self.dim)  # Adjusted noise type
                result = minimize(func, perturbed_point, method='Nelder-Mead', options={
                    'maxfev': self.budget - self.evaluations,
                    'fatol': 1e-8, 'xatol': 1e-8})  # Further reduced tolerance
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                    adaptive_factor *= 0.9  # More aggressive adaptive factor reduction
                self.evaluations += result.nfev

            # Stochastic reinitialization with adaptive learning rate
            if np.random.rand() < 0.1:  # Random restart probability
                initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
            else:
                initial_points = np.random.uniform(np.maximum(bounds[:, 0], best_point - adaptive_factor),
                                                   np.minimum(bounds[:, 1], best_point + adaptive_factor),
                                                   (self.dim + 1, self.dim))

        return best_point