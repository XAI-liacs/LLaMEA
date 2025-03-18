import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        num_initial_samples = max(5, (self.budget - self.evaluations) // 4)  # Changed from 6 to 4
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for sample in samples:
            res = minimize(func, sample, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            if self.evaluations >= self.budget:
                break

        while self.evaluations < self.budget:
            new_lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
            new_ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))
            
            res = minimize(func, best_solution, bounds=list(zip(new_lb, new_ub)), method='L-BFGS-B')
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            if abs(res.fun - best_value) < 1e-7:  # Tightened from 1e-6 to 1e-7
                break  # Added break to enhance stopping criteria

        return best_solution