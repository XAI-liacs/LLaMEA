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
        
        # Step 1: Initial Sampling
        num_samples = min(self.budget // 4, 12)  # Allocate up to a quarter of the budget
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
                grad[i] = (limited_func(x1) - limited_func(x2)) / (2 * epsilon)
            return grad
        
        # Adaptive sampling loop with gradient guidance
        for _ in range(3):
            gradient = estimate_gradient(best_sample)
            step_size = 0.5 / (1.0 + norm(gradient))  # Modified step size
            new_sample = best_sample - step_size * gradient
            new_sample = np.clip(new_sample, lb, ub)
            new_eval = limited_func(new_sample)
            if new_eval < func(best_sample):
                best_sample = new_sample
        
        # Step 3: Final Local Optimization using BFGS
        result = minimize(limited_func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        
        return result.x

# Usage example with a hypothetical black box function
# optimizer = AdaptiveGradientSampling(budget=50, dim=2)
# best_params = optimizer(black_box_func)