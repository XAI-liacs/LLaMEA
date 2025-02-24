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

        gradient_strength = 0.5  # New variable for gradient strength

        while self.evaluations < self.budget:
            for point in initial_points:
                if self.evaluations >= self.budget:
                    break
                result = minimize(func, point, method='Nelder-Mead', options={
                    'maxfev': self.budget - self.evaluations,
                    'fatol': 1e-6, 'xatol': 1e-6})  
                if result.fun < best_value:
                    best_value = result.fun
                    best_point = result.x
                    adaptive_factor *= 0.95  
                self.evaluations += result.nfev

            # Integrate gradient information for enhanced refinement
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                shift = np.zeros(self.dim)
                shift[i] = 1e-5
                grad[i] = (func(best_point + shift) - func(best_point)) / 1e-5
            best_point = best_point - gradient_strength * grad  

            # Dual-phase adaptive initialization
            if self.evaluations < self.budget / 2:
                initial_points = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.dim + 1, self.dim))
            else:
                initial_points = np.random.uniform(np.maximum(bounds[:, 0], best_point - adaptive_factor),
                                                   np.minimum(bounds[:, 1], best_point + adaptive_factor),
                                                   (self.dim + 1, self.dim))

        return best_point