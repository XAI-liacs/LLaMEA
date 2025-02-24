import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def uniform_sampling(self, bounds, num_samples):
        samples = []
        for lb, ub in zip(bounds.lb, bounds.ub):
            samples.append(np.random.uniform(lb, ub, num_samples))
        return np.array(samples).T

    def local_optimization(self, func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, result.fun

    def __call__(self, func):
        num_initial_samples = min(10, self.budget // 2)
        initial_points = self.uniform_sampling(func.bounds, num_initial_samples)
        
        best_solution = None
        best_value = float('inf')

        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        for x0 in initial_points:
            if self.budget <= 0:
                break

            x, value = self.local_optimization(func, x0, bounds)
            self.budget -= 1  # Counting the local optimization as a single budget usage

            if value < best_value:
                best_value = value
                best_solution = x

        return best_solution