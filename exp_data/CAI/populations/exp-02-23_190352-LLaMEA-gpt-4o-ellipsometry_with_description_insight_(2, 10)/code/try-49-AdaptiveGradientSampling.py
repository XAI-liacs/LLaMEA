import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm

class AdaptiveGradientSampling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Extract bounds
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Step 1: Initial Sampling with expanded exploration
        num_samples = min(self.budget // 3, 15)  # Adjusted initial sample allocation
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])
        
        # Find the best initial sample
        best_index = np.argmin(sample_evals)
        best_sample = samples[best_index]
        
        # Step 2: Gradient Estimation and Adaptive Sampling
        remaining_budget = self.budget - num_samples
        eval_count = 0
        
        def limited_func(x):
            nonlocal eval_count
            if eval_count >= remaining_budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return func(x)
        
        def estimate_gradient(x, epsilon=1e-5):
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                x1, x2 = np.array(x), np.array(x)
                x1[i] += epsilon
                x2[i] -= epsilon
                grad[i] = (limited_func(x1) - limited_func(x2)) / (2 * epsilon)  # Changed from central to centralized difference
            return grad
        
        # Adaptive sampling loop with gradient guidance
        for _ in range(4):  # Increased adaptive iterations
            gradient = estimate_gradient(best_sample)
            step_size = 0.7 / (1.0 + norm(gradient))  # Adjusted step size for faster convergence
            new_sample = best_sample - step_size * gradient
            new_sample = np.clip(new_sample, lb, ub)
            new_eval = limited_func(new_sample)
            if new_eval < func(best_sample):
                best_sample = new_sample
            
            # Dynamic adjustment of sampling based on current progress
            if eval_count < remaining_budget // 2:
                samples = np.random.uniform(lb, ub, (3, self.dim))
                sample_evals = np.array([limited_func(sample) for sample in samples])
                best_idx = np.argmin(sample_evals)
                if sample_evals[best_idx] < func(best_sample):
                    best_sample = samples[best_idx]
        
        # Step 3: Final Local Optimization using BFGS
        result = minimize(limited_func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        
        return result.x