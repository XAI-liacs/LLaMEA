import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        num_samples = max(5, self.budget // 3)
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])
        
        best_five_indices = np.argsort(sample_evals)[:5]
        # Refine using covariance-based sampling
        best_samples = samples[best_five_indices]
        cov_matrix = np.cov(best_samples, rowvar=False)  # Calculate covariance
        refined_samples = best_samples + np.random.multivariate_normal(np.zeros(self.dim), cov_matrix, size=5)
        refined_evals = np.array([func(refined_sample) for refined_sample in refined_samples])
        
        best_index = np.argmin(refined_evals)
        best_sample = refined_samples[best_index]
        
        remaining_budget = self.budget - num_samples - len(refined_samples)
        eval_count = 0
        
        def limited_func(x):
            nonlocal eval_count
            if eval_count >= remaining_budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return func(x)
        
        result = minimize(limited_func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        
        return result.x