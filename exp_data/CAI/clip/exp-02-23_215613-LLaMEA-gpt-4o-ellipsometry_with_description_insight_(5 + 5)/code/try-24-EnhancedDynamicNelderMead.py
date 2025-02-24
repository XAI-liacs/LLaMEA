import numpy as np
from scipy.optimize import minimize

class EnhancedDynamicNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def stochastic_gradient_step(self, point, func):
        """Apply a small stochastic gradient-like step to exploit the local landscape."""
        epsilon = 1e-8  # Small step size for finite difference
        grad = np.zeros(self.dim)
        base_value = func(point)
        for i in range(self.dim):
            step = np.zeros(self.dim)
            step[i] = epsilon
            grad[i] = (func(point + step) - base_value) / epsilon
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break
        return point - 0.01 * grad  # Adjust step size as needed

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
        best_point = initial_points[0]
        best_value = float('inf')

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                result = minimize(func, point, method='Nelder-Mead', options={'maxfev': self.budget - self.evaluations, 'adaptive': True, 'fatol': 1e-6})
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                self.evaluations += result.nfev
                
                # Apply stochastic gradient step if budget allows
                if self.evaluations < self.budget:
                    new_point = self.stochastic_gradient_step(result.x, func)
                    new_value = func(new_point)
                    if new_value < best_value:
                        best_value = new_value
                        best_point = new_point
                    self.evaluations += 1

            # Adaptive initial sampling for the next iteration
            step_size = (self.budget - self.evaluations) / self.budget
            initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim)) + step_size * np.random.uniform(-0.5, 0.5, (self.dim + 1, self.dim))

        return best_point