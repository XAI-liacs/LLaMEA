import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 5, 25)  # Increased initial sampling for better global navigation
        
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

        # Adaptive refinement step using a mixture of local optimizers based on remaining budget
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            evaluations += 1
            return func(x)

        if evaluations < self.budget:
            remaining_budget = self.budget - evaluations

            # Determine the optimization method based on remaining budget
            if remaining_budget > self.budget // 4:
                method = 'L-BFGS-B'
            else:
                method = 'Nelder-Mead'

            result = minimize(wrapped_func, best_solution, method=method, bounds=bounds if method == 'L-BFGS-B' else None)
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun

        return best_solution