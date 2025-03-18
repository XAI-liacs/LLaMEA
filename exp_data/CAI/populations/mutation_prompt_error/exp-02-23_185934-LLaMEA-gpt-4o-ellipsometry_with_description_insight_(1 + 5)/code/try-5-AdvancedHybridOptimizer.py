import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_samples = min(10, budget // 2)
        self.multi_start_points = max(2, budget // 10)

    def __call__(self, func):
        # Adaptive sampling for initial exploration
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Multi-start strategy for better initial sample distribution
        for _ in range(self.multi_start_points):
            for _ in range(self.initial_samples):
                x0 = np.random.uniform(lb, ub, self.dim)
                value = func(x0)
                self.evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = x0

                if self.evaluations >= self.budget:
                    return best_solution

        # Local optimization using Nelder-Mead
        remaining_budget = self.budget - self.evaluations
        if remaining_budget > 0:
            # Perform multiple runs of local optimization from different starting points
            for _ in range(self.multi_start_points):
                res = minimize(func, best_solution, method='Nelder-Mead', 
                               options={'maxiter': remaining_budget // self.multi_start_points, 'adaptive': True})
                if res.fun < best_value:
                    best_value = res.fun
                    best_solution = res.x

        return best_solution