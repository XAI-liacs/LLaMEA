import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 4, 10)
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        evaluations = 0

        # Evaluate initial samples
        for sample in initial_samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Adaptive Sampling based on variance reduction
        while evaluations < self.budget:
            sampled_solutions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
            sampled_values = []
            for sample in sampled_solutions:
                if evaluations >= self.budget:
                    break
                value = func(sample)
                evaluations += 1
                sampled_values.append(value)
            variance = np.var(sampled_values)
            if variance < 1e-5:
                break
            best_idx = np.argmin(sampled_values)
            if sampled_values[best_idx] < best_value:
                best_value = sampled_values[best_idx]
                best_solution = sampled_solutions[best_idx]

        # Refine the best initial sample using BFGS
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            evaluations += 1
            return func(x)
        
        if evaluations < self.budget:
            result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun

        return best_solution