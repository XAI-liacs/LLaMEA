import numpy as np
from scipy.optimize import minimize

class AdaptiveGradientOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Extract bounds
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Step 1: Adaptive Sampling
        num_samples = min(self.budget // 4, 20)  # Use up to a fourth of the budget, increased to 20 samples
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])
        
        # Adaptive selection based on performance
        top_indices = np.argsort(sample_evals)[:5]
        top_samples = samples[top_indices]
        
        # Calculate centroid for initial guess
        initial_guess = np.mean(top_samples, axis=0)
        
        # Calculate the variance and adjust step sizes
        variances = np.var(top_samples, axis=0)
        step_sizes = np.maximum(variances, np.full(self.dim, 1e-3))
        
        # Use remaining budget for local optimization
        remaining_budget = self.budget - num_samples
        eval_count = 0
        
        def limited_func(x):
            nonlocal eval_count
            if eval_count >= remaining_budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return func(x)
        
        # Step 2: Local Optimization using BFGS with adaptive step sizes
        result = minimize(limited_func, initial_guess, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'eps': step_sizes})
        
        return result.x

# Usage example with a hypothetical black box function
# optimizer = AdaptiveGradientOptimizer(budget=50, dim=2)
# best_params = optimizer(black_box_func)