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
        gradient_step = adaptive_factor * 0.01  # Small step for gradient-based refinement

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                result = minimize(func, point, method='Nelder-Mead', options={
                    'maxfev': self.budget - self.evaluations,
                    'fatol': 1e-7, 'xatol': 1e-7})
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                    adaptive_factor *= 0.95
                self.evaluations += result.nfev

            # Adaptive reset with refined search space based on best point
            initial_points = np.random.uniform(np.maximum(bounds[:, 0], best_point - adaptive_factor),
                                               np.minimum(bounds[:, 1], best_point + adaptive_factor),
                                               (self.dim + 1, self.dim))

            # Gradient-based refinement
            if self.evaluations < self.budget:
                grad_estimate = np.zeros(self.dim)
                for i in range(self.dim):
                    perturb = np.zeros(self.dim)
                    perturb[i] = gradient_step[i]
                    f_plus = func(best_point + perturb)
                    f_minus = func(best_point - perturb)
                    grad_estimate[i] = (f_plus - f_minus) / (2 * gradient_step[i])
                    self.evaluations += 2  # Count two evaluations

                refined_point = best_point - gradient_step * grad_estimate
                refined_point = np.clip(refined_point, bounds[:, 0], bounds[:, 1])
                refined_value = func(refined_point)
                self.evaluations += 1

                if refined_value < best_value:
                    best_value = refined_value
                    best_point = refined_point

        return best_point