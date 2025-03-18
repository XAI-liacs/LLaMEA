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
            new_range = (ub - lb) / 4  # More precise exploration
            new_lb = max(lb, middle - new_range / 4)
            new_ub = min(ub, middle + new_range / 4)
            if new_lb >= new_ub: 
                new_lb, new_ub = lb, ub
            new_bounds.append((new_lb, new_ub))
        return new_bounds

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        result = minimize(func, initial_guess, method='Nelder-Mead', options={'maxfev': self.budget//4})  # Adjusted allocation
        
        self.evaluations += result.nfev
        if self.evaluations >= self.budget:
            return result.x
        
        current_best = result.x
        adaptive_bounds = self._adaptive_bounds(bounds, current_best)
        
        random_shift = np.random.uniform(-0.01, 0.01, self.dim)  # Tighter random shift range
        
        adaptive_initial_guess = np.clip(
            current_best + random_shift, 
            [ab[0] for ab in adaptive_bounds], 
            [ab[1] for ab in adaptive_bounds]
        )
        
        method = 'BFGS' if self.evaluations < self.budget * 0.7 else 'Nelder-Mead'
        remaining_budget = self.budget - self.evaluations
        method_budget = remaining_budget // 2 if method == 'BFGS' else remaining_budget
        
        if self.evaluations % (self.budget // 6) == 0:
            adaptive_initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        
        result = minimize(func, adaptive_initial_guess, bounds=adaptive_bounds, method=method, options={'maxfev': method_budget})
        
        if result.fun < 1e-6:  # Introduced restart mechanism if near global optimum
            self.evaluations = 0
        
        return result.x