import numpy as np
from scipy.optimize import minimize

class EnhancedPhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Calculate the number of initial samples, ensuring diversity
        num_initial_samples = min(self.budget // 3, 15)

        # Generate initial samples using Latin Hypercube Sampling for better diversity
        initial_samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        # Placeholder for the best solution and its evaluation
        best_solution = None
        best_value = float('inf')
        
        # Budget tracking
        evaluations_used = 0
        
        # Initial evaluations
        for sample in initial_samples:
            if evaluations_used >= self.budget:
                break
            result = minimize(func, sample, method='BFGS', bounds=np.array(list(zip(lb, ub))))
            evaluations_used += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        # Further refinement
        while evaluations_used < self.budget:
            # Dynamically adjust the bounds based on the current best solution
            # Narrow bounds as a function of the remaining evaluations
            adjust_factor = max(0.01, 1 - evaluations_used / self.budget)
            current_bounds = [(max(lb[i], best_solution[i] - adjust_factor * (ub[i] - lb[i])), 
                               min(ub[i], best_solution[i] + adjust_factor * (ub[i] - lb[i]))) 
                              for i in range(self.dim)]
            
            # Run optimization from the best solution with adjusted bounds
            result = minimize(func, best_solution, method='BFGS', bounds=current_bounds)
            evaluations_used += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution