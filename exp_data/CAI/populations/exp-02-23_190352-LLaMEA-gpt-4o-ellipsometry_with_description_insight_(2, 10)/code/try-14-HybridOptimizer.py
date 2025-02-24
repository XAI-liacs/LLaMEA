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
        
        # Step 1: Adaptive Sampling
        initial_samples = min(max(self.budget // 5, 5), 20)  # Adaptive range based on budget
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])
        
        # Find the best initial sample
        best_index = np.argmin(sample_evals)
        best_sample = samples[best_index]
        
        # Use a weighted average of top samples as initial guess
        sorted_indices = np.argsort(sample_evals)
        top_samples = samples[sorted_indices[:3]]
        best_sample = np.average(top_samples, axis=0, weights=[0.5, 0.3, 0.2])
        
        # Step 2: Dynamic Local Refinement
        def dynamic_bounds_adjustment(x):
            # Expand search around current best estimate
            epsilon = 0.05 * (ub - lb)
            new_lb = np.maximum(lb, x - epsilon)
            new_ub = np.minimum(ub, x + epsilon)
            return list(zip(new_lb, new_ub))
        
        # Use remaining budget for BFGS optimization
        remaining_budget = self.budget - initial_samples
        eval_count = 0
        
        def limited_func(x):
            nonlocal eval_count
            if eval_count >= remaining_budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return func(x)
        
        # Local Optimization using BFGS with dynamic bounds
        bounds = dynamic_bounds_adjustment(best_sample)
        result = minimize(limited_func, best_sample, method='L-BFGS-B', bounds=bounds)
        
        return result.x

# Example usage with a hypothetical black box function
# optimizer = HybridOptimizer(budget=50, dim=2)
# best_params = optimizer(black_box_func)