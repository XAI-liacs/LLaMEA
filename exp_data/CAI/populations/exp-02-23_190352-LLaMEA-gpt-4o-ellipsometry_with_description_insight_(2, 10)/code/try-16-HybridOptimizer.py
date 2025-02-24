import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Step 1: Dynamic Uniform Sampling based on budget
        num_samples = max(3, int(self.budget * 0.2))  # 20% of the budget for initial sampling
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])
        
        # Adaptive initialization using top samples' weighted mean
        sorted_indices = np.argsort(sample_evals)
        top_samples = samples[sorted_indices[:5]]
        weights = np.linspace(1, 0.2, num=5)
        best_sample = np.average(top_samples, axis=0, weights=weights)
        
        remaining_budget = self.budget - num_samples
        eval_count = 0
        
        def limited_func(x):
            nonlocal eval_count
            if eval_count >= remaining_budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return func(x)
        
        # Step 2: Local Optimization using BFGS with improved initialization
        result = minimize(limited_func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        
        return result.x