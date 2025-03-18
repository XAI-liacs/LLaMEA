import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        # Adjust initial sampling size based on remaining budget (reduce multiplication factor from 8 to 6)
        num_initial_samples = max(1, (self.budget - self.evaluations) // 6)
        initial_samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_initial_samples, self.dim))

        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            # Introduce adaptive local search method based on evaluations
            if self.evaluations < 0.5 * self.budget:
                solution, value = self.local_search(func, sample, bounds)
            else:
                solution, value = self.gradient_refinement(func, sample, bounds)
            if value < best_value:
                best_solution, best_value = solution, value

        return best_solution

    def local_search(self, func, initial_point, bounds):
        if self.evaluations >= self.budget:
            return initial_point, func(initial_point)

        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
        self.evaluations += result.nfev

        return result.x, result.fun

    def gradient_refinement(self, func, initial_point, bounds):
        # Use a different optimizer (Nelder-Mead) for final refinement
        result = minimize(func, initial_point, method='Nelder-Mead', bounds=bounds, options={'maxfev': self.budget - self.evaluations})
        self.evaluations += result.nfev

        return result.x, result.fun