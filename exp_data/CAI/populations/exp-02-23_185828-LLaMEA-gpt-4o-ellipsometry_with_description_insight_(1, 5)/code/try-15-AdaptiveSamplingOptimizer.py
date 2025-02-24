import numpy as np
from scipy.optimize import minimize

class AdaptiveSamplingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 3, 20)  # Start with fewer initial samples

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

        # Adaptive sampling near the best found solution
        for _ in range(self.budget // 3):
            if evaluations >= self.budget:
                break
            perturbation = np.random.uniform(-0.05, 0.05, self.dim)
            candidate_sample = np.clip(best_solution + perturbation, bounds[:, 0], bounds[:, 1])
            value = func(candidate_sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = candidate_sample
        
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