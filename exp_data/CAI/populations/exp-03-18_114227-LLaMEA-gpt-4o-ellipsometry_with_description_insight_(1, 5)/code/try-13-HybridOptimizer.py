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
            new_lb = max(lb, middle - new_range / 3)
            new_ub = min(ub, middle + new_range / 3)
            if new_lb >= new_ub:
                new_lb, new_ub = lb, ub
            new_bounds.append((new_lb, new_ub))
        return new_bounds

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        best_solution = None
        best_cost = float('inf')
        
        for _ in range(3):  # Introduce random restarts
            initial_guess = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            result = minimize(func, initial_guess, method='Nelder-Mead', options={'maxfev': self.budget//3})
            self.evaluations += result.nfev
            if result.fun < best_cost:
                best_cost = result.fun
                best_solution = result.x
            if self.evaluations >= self.budget:
                return best_solution

        current_best = best_solution
        adaptive_bounds = self._adaptive_bounds(bounds, current_best)
        
        adaptive_initial_guess = np.clip(
            current_best + np.random.uniform(-0.1, 0.1, self.dim), 
            [ab[0] for ab in adaptive_bounds], 
            [ab[1] for ab in adaptive_bounds]
        )
        
        result = minimize(func, adaptive_initial_guess, bounds=adaptive_bounds, method='Nelder-Mead', options={'maxfev': self.budget - self.evaluations})
        
        return result.x