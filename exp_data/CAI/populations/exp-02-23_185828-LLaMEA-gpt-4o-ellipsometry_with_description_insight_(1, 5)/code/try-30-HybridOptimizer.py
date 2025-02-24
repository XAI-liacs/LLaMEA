import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 4, 30)  # Reduced initial sampling size

        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        evaluations = 0

        for sample in initial_samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            evaluations += 1
            return func(x)
        
        while evaluations < self.budget:
            perturbation_std = 0.01  # Constant perturbation
            best_solution += np.random.normal(0, perturbation_std, self.dim)
            result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun
            if evaluations + self.dim < self.budget:  # Adaptive restart condition
                initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (5, self.dim))
                for sample in initial_samples:
                    value = func(sample)
                    evaluations += 1
                    if value < best_value:
                        best_value = value
                        best_solution = sample

        return best_solution