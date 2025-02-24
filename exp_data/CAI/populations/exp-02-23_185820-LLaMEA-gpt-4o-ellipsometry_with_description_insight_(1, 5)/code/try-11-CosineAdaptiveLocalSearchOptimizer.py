import numpy as np
from scipy.optimize import minimize
import math

class CosineAdaptiveLocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        
        # Step 1: Initial uniform sampling for a broad exploration
        num_samples = min(10, self.budget // 3)
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_samples, self.dim))
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        # Step 2: Select the best initial guess
        best_idx = np.argmin(initial_values)
        best_guess = initial_samples[best_idx]
        
        result = None
        while self.evals < self.budget:
            # Step 3: Cosine annealing for dynamically shrinking bounds
            cosine_factor = (1 + math.cos(math.pi * self.evals / self.budget)) / 2
            scale_factor = 0.3 * cosine_factor
            narrowed_bounds = [
                (max(lb, best_guess[i] - scale_factor * (ub - lb)), min(ub, best_guess[i] + scale_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            # Local search with narrowed bounds using BFGS
            result = minimize(
                func,
                best_guess,
                method='L-BFGS-B',
                bounds=narrowed_bounds,
                options={'maxfun': min(10, self.budget - self.evals)}
            )
            
            # Update evaluations and best guess
            self.evals += result.nfev
            if result.fun < func(best_guess):
                best_guess = result.x

        return best_guess