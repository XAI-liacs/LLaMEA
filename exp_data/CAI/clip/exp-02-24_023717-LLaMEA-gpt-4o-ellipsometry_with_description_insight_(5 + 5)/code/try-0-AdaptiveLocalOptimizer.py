import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        self.evaluations = 0  # Reset evaluations counter
        bounds = func.bounds
        initial_points = self.uniform_sampling(bounds, 10)
        best_solution = None
        best_score = np.inf

        for point in initial_points:
            res = self.optimize_from_point(func, point, bounds)
            if res.fun < best_score:
                best_solution = res.x
                best_score = res.fun

            if self.evaluations >= self.budget:
                break

        return best_solution, best_score

    def uniform_sampling(self, bounds, num_samples):
        lb, ub = bounds.lb, bounds.ub
        samples = [lb + np.random.rand(self.dim) * (ub - lb) for _ in range(num_samples)]
        return samples

    def optimize_from_point(self, func, start_point, bounds):
        result = minimize(
            self.evaluate_function,
            start_point,
            method='L-BFGS-B',
            bounds=list(zip(bounds.lb, bounds.ub)),
            options={'disp': False}
        )
        return result

    def evaluate_function(self, x):
        if self.evaluations < self.budget:
            self.evaluations += 1
            return func(x)
        else:
            raise RuntimeError("Budget exhausted")