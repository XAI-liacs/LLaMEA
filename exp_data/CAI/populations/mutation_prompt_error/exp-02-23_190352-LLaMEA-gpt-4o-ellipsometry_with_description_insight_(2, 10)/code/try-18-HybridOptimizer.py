import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Extract bounds
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Step 1: Dynamic Sampling
        num_samples = max(5, self.budget // 4)  # Adjusted to use up to a quarter of the budget
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])
        
        # Find the best initial sample
        best_index = np.argmin(sample_evals)
        best_sample = samples[best_index]
        
        # Use remaining budget for BFGS optimization
        remaining_budget = self.budget - num_samples
        eval_count = 0
        
        def limited_func(x):
            nonlocal eval_count
            if eval_count >= remaining_budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return func(x)
        
        # Step 2: Local Optimization using BFGS with adaptive step size
        result = minimize(limited_func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxiter': max(2, remaining_budget // 2)})
        
        return result.x

# Usage example with a hypothetical black box function
# optimizer = HybridOptimizer(budget=50, dim=2)
# best_params = optimizer(black_box_func)