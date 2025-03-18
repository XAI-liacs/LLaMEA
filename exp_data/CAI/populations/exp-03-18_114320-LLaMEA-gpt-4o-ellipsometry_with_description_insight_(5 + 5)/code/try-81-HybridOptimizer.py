import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Unpack bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        remaining_budget = self.budget
        
        # Initial exploration with stratified sampling for wider coverage
        num_samples = min(remaining_budget // 3, 20 * self.dim)
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget -= num_samples
        
        # Evaluate samples and find the best initial guess
        sample_evaluations = np.array([func(sample) for sample in samples])
        best_idx = np.argmin(sample_evaluations)
        best_solution = samples[best_idx]
        best_value = sample_evaluations[best_idx]
        
        # Local optimization using a hybrid BFGS and Nelder-Mead approach
        def local_objective(x):
            return func(x)
        
        # Adaptive narrowing of bounds based on initial exploration results
        adaptive_bounds = [(max(lb[i], best_solution[i] - 0.1 * (ub[i] - lb[i])), 
                            min(ub[i], best_solution[i] + 0.1 * (ub[i] - lb[i]))) for i in range(self.dim)]
        
        # Adjust budget for local optimization
        remaining_budget = max(remaining_budget, 5)
        result = minimize(local_objective, best_solution, bounds=adaptive_bounds, method='Nelder-Mead', options={'maxfev': remaining_budget})
        
        if result.fun < best_value:
            best_solution = result.x
            best_value = result.fun
        
        return best_solution