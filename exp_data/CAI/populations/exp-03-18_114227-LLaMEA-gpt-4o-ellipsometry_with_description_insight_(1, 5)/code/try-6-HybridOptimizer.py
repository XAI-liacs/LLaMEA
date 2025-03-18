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
            new_range = (ub - lb) / 4
            new_lb = max(lb, middle - new_range / 3)  # Refined adjustment
            new_ub = min(ub, middle + new_range / 3)  # Refined adjustment
            if new_lb >= new_ub:
                new_lb, new_ub = lb, ub
            new_bounds.append((new_lb, new_ub))
        return new_bounds

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Strategic initial sampling for diverse starting point
        initial_guess = np.mean([func.bounds.lb, func.bounds.ub], axis=0) + np.random.uniform(-0.05, 0.05, self.dim)
        
        result = minimize(func, initial_guess, method='Nelder-Mead', options={'maxfev': self.budget//2})
        
        self.evaluations += result.nfev
        if self.evaluations >= self.budget:
            return result.x
        
        current_best = result.x
        adaptive_bounds = self._adaptive_bounds(bounds, current_best)
        
        adaptive_initial_guess = np.clip(
            current_best + np.random.uniform(-0.1, 0.1, self.dim), 
            [ab[0] for ab in adaptive_bounds], 
            [ab[1] for ab in adaptive_bounds]
        )
        
        result = minimize(func, adaptive_initial_guess, bounds=adaptive_bounds, method='Nelder-Mead', options={'maxfev': self.budget - self.evaluations})
        
        return result.x