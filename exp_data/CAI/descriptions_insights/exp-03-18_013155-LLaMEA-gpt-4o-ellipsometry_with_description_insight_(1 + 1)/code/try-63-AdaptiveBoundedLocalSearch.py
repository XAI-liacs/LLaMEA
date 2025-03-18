import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Uniform sampling for initial guesses, dynamically adjusted based on remaining budget
        num_initial_samples = max(5, (self.budget - self.evaluations) // 4)  
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        previous_best_value = float('inf')  
        
        # Begin search
        for sample in samples:
            # Local optimization using L-BFGS-B
            res = minimize(func, sample, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            if self.evaluations >= self.budget:
                break

        # Iteratively adjust bounds and refine
        while self.evaluations < self.budget:
            # Narrow the bounds based on current best solution
            new_lb = np.maximum(lb, best_solution - 0.05 * (ub - lb))  # Adjusted from 0.1 to 0.05
            new_ub = np.minimum(ub, best_solution + 0.05 * (ub - lb))  # Adjusted from 0.1 to 0.05
            
            # Local optimization within adjusted bounds
            res = minimize(func, best_solution, bounds=list(zip(new_lb, new_ub)), method='L-BFGS-B')
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            # Selective termination based on cost function plateau detection
            if abs(previous_best_value - best_value) < 1e-6:  
                break
            previous_best_value = best_value  

        return best_solution