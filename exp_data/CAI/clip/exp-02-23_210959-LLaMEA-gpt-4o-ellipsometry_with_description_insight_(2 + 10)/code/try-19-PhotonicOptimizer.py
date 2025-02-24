import numpy as np
from scipy.optimize import minimize

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Calculate the maximum number of initial samples to generate
        num_initial_samples = min(self.budget // 2, 15)  # Increased initial samples
        
        # Generate random initial samples uniformly within bounds
        initial_samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        # Placeholder for the best solution and its evaluation
        best_solution = None
        best_value = float('inf')
        
        # Initial evaluations
        for sample in initial_samples:
            result = minimize(func, sample, method='L-BFGS-B', bounds=np.array(list(zip(lb, ub))), options={'eps': 1e-3})  # Adaptive step size
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        # Further evaluations within the remaining budget
        remaining_budget = self.budget - num_initial_samples
        while remaining_budget > 0:
            # Dynamically adjust the bounds based on the current best solution
            current_bounds = [(max(lb[i], best_solution[i] - 0.1 * (ub[i] - lb[i])), min(ub[i], best_solution[i] + 0.1 * (ub[i] - lb[i]))) for i in range(self.dim)]
            
            # Run optimization from the best solution with adjusted bounds
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=current_bounds)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            remaining_budget -= 1
        
        return best_solution