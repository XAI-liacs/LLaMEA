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
            new_range = (ub - lb) / 4  # Changed to 4 for a balanced exploration-exploitation
            new_lb = max(lb, middle - new_range / 2)  # Adjusted from /3 to /2 for more stability
            new_ub = min(ub, middle + new_range / 2)  # Adjusted from /3 to /2 for more stability
            if new_lb >= new_ub: 
                new_lb, new_ub = lb, ub
            new_bounds.append((new_lb, new_ub))
        return new_bounds

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        result = minimize(func, initial_guess, method='Nelder-Mead', options={'maxfev': self.budget//3})

        self.evaluations += result.nfev
        if self.evaluations >= self.budget:
            return result.x
        
        current_best = result.x
        adaptive_bounds = self._adaptive_bounds(bounds, current_best)
        
        random_shift = np.random.uniform(-0.03, 0.03, self.dim)  # Adjusted random shift range for precision
        adaptive_initial_guess = np.clip(
            current_best + random_shift, 
            [ab[0] for ab in adaptive_bounds], 
            [ab[1] for ab in adaptive_bounds]
        )
        
        method = 'Nelder-Mead' if self.evaluations < self.budget * 0.7 else 'BFGS'  # Switched order of method use
        
        if self.evaluations % (self.budget // 6) == 0:
            adaptive_initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        result = minimize(func, adaptive_initial_guess, bounds=adaptive_bounds, method=method, options={'maxfev': self.budget - self.evaluations})
        
        return result.x