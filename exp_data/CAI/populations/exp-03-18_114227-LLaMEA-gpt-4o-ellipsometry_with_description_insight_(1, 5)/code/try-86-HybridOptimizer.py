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
            new_range = (ub - lb) / 3
            new_lb = max(lb, middle - new_range / 3)
            new_ub = min(ub, middle + new_range / 3)
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
        
        temperature = max(0.01, 1.0 - self.evaluations / self.budget)
        random_shift = np.random.uniform(-0.03, 0.03, self.dim) * temperature * np.linalg.norm(result.jac)  # Adjusted shift
        
        adaptive_initial_guess = np.clip(
            np.random.uniform([ab[0] for ab in adaptive_bounds], [ab[1] for ab in adaptive_bounds]),  
            [ab[0] for ab in adaptive_bounds], 
            [ab[1] for ab in adaptive_bounds]
        )
        
        if self.evaluations < self.budget * 0.5: method = 'BFGS'  
        remaining_budget = self.budget - self.evaluations
        method_budget = remaining_budget // 2 if method == 'BFGS' else remaining_budget
        
        if self.evaluations % (self.budget // 6) == 0 or np.linalg.norm(result.fun) > 1e-2:  
            adaptive_initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        result = minimize(func, adaptive_initial_guess, bounds=adaptive_bounds, method=method, options={'maxfev': method_budget})
        
        return result.x