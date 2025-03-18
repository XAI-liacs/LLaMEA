import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Unpack bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        # Remaining budget after initial exploration
        remaining_budget = self.budget
        
        # Initial exploration with uniform sampling
        num_samples = min(remaining_budget // 2, 15 * self.dim)  # Increase samples for better diversity
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget -= num_samples
        
        # Evaluate samples and find the best initial guess
        sample_evaluations = np.array([func(sample) for sample in samples])
        best_idx = np.argmin(sample_evaluations * np.linspace(1.0, 0.5, num_samples))  # Dynamic evaluation weight
        best_solution = samples[best_idx]
        best_value = sample_evaluations[best_idx]
        
        # Local optimization using BFGS or Nelder-Mead based on landscape smoothness
        def local_objective(x):
            return func(x)
        
        # Adjust budget for local optimization based on initial evaluations
        remaining_budget = max(remaining_budget, 5)
        local_method = 'Nelder-Mead' if np.std(sample_evaluations) < 0.1 else 'L-BFGS-B'  # Adaptive method switch
        result = minimize(local_objective, best_solution, bounds=list(zip(lb, ub)), method=local_method, options={'maxfun': remaining_budget})
        
        # Choose the best solution found
        if result.fun < best_value:
            best_solution = result.x
            best_value = result.fun
        
        return best_solution