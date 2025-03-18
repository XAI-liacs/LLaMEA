import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def _adaptive_bounds(self, bounds, current_best):
        new_bounds = []
        for i in range(self.dim):
            lb, ub = bounds[i]
            middle = current_best[i]
            new_range = (ub - lb) / 5  # Changed division factor for tighter bounds
            new_lb = max(lb, middle - new_range / 4)  # Adjusted for narrower search space
            new_ub = min(ub, middle + new_range / 4)  # Adjusted for narrower search space
            if new_lb >= new_ub:
                new_lb, new_ub = lb, ub
            new_bounds.append((new_lb, new_ub))
        return new_bounds

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget//3}) # Changed optimizer and method

        self.evaluations += result.nfev
        if self.evaluations >= self.budget:
            return result.x
        
        current_best = result.x
        adaptive_bounds = self._adaptive_bounds(bounds, current_best)
        
        adaptive_initial_guess = np.clip(
            current_best + np.random.uniform(-0.05, 0.05, self.dim),  # Narrower random perturbation
            [ab[0] for ab in adaptive_bounds], 
            [ab[1] for ab in adaptive_bounds]
        )
        
        result = minimize(func, adaptive_initial_guess, bounds=adaptive_bounds, method='L-BFGS-B', options={'maxfun': self.budget - self.evaluations}) # Changed optimizer and method
        
        return result.x