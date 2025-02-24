import numpy as np
from scipy.optimize import minimize

class DynamicNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def estimate_gradient(self, func, point, epsilon=1e-8):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            point_high = np.copy(point)
            point_low = np.copy(point)
            point_high[i] += epsilon
            point_low[i] -= epsilon
            gradient[i] = (func(point_high) - func(point_low)) / (2 * epsilon)
            self.evaluations += 2  # Count gradient evaluations
        return gradient

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
        best_point = initial_points[0]
        best_value = float('inf')

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                grad = self.estimate_gradient(func, point)
                result = minimize(func, point, method='Nelder-Mead', jac=grad, options={'maxfev': self.budget - self.evaluations})
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                self.evaluations += result.nfev

            # Reset with new random initial points for broad exploration
            initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))

        return best_point