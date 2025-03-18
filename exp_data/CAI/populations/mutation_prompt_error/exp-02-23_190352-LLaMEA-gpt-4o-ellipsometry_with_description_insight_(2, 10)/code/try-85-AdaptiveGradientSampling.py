import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm

class AdaptiveGradientSampling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        num_samples = min(self.budget // 2, 25)
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])
        
        best_index = np.argmin(sample_evals)
        best_sample = samples[best_index]
        
        remaining_budget = self.budget - num_samples
        eval_count = 0
        
        def limited_func(x):
            nonlocal eval_count
            if eval_count >= remaining_budget:
                raise ValueError("Budget exceeded")
            eval_count += 1
            return func(x)
        
        def estimate_gradient(x, epsilon=1e-6):  # Adjusted epsilon value for better precision
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                x1, x2 = np.array(x), np.array(x)
                x1[i] += epsilon
                x2[i] -= epsilon
                grad[i] = (limited_func(x1) - limited_func(x2)) / (2 * epsilon)
            return grad
        
        for _ in range(5):  # Increased adaptive iterations
            gradient = estimate_gradient(best_sample)
            step_size = 0.8 / (1.0 + 0.3 * norm(gradient))  # Refined step size calculation
            new_sample = best_sample - step_size * gradient
            new_sample = np.clip(new_sample, lb, ub)
            new_eval = limited_func(new_sample)
            if new_eval < func(best_sample):
                best_sample = new_sample
            
            if eval_count < remaining_budget // 2:
                samples = np.random.uniform(lb, ub, (4, self.dim))  # Increased sample size
                sample_evals = np.array([limited_func(sample) for sample in samples])
                best_idx = np.argmin(sample_evals)
                if sample_evals[best_idx] < func(best_sample):
                    best_sample = samples[best_idx]
        
        result = minimize(limited_func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        
        return result.x