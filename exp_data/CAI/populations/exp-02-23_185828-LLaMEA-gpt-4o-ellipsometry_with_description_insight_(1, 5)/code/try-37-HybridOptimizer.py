import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 2, 50)  # Modified initial sampling size

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
        
        # Refine the best initial sample using L-BFGS-B
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            evaluations += 1
            return func(x)
        
        if evaluations < self.budget:
            perturbation_std = 0.005 + 0.02 * np.std(initial_samples, axis=0).mean()  # Enhanced perturbation
            best_solution += np.random.normal(0, perturbation_std, self.dim)
            # Added constraint handling
            result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-9})
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun

        return best_solution