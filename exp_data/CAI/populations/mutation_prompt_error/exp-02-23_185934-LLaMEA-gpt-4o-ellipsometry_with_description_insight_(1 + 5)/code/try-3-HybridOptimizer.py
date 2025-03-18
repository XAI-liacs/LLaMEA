import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_samples = min(10, int(budget * 0.3))  # Changed from budget // 2 to int(budget * 0.3)

    def __call__(self, func):
        # Uniform sampling for initial exploration
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        
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
            res = minimize(func, best_solution, method='Nelder-Mead', bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': remaining_budget, 'adaptive': True})
            best_solution = res.x if res.fun < best_value else best_solution

        return best_solution