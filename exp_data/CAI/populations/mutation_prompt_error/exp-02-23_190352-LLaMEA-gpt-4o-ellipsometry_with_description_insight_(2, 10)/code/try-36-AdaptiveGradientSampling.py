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
        
        num_samples = min(self.budget // 4, 12)
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
        
        def estimate_gradient(x, epsilon=1e-5):
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                x1, x2 = np.array(x), np.array(x)
                x1[i] += epsilon
                x2[i] -= epsilon
                grad[i] = (limited_func(x1) - limited_func(x2)) / (2 * epsilon)
            return grad
        
        for _ in range(2): # Reduced iterations for increased exploration
            gradient = estimate_gradient(best_sample)
            step_size = 0.3 / (1.0 + norm(gradient)) # Adjusted step size
            new_sample = best_sample - step_size * gradient
            new_sample = np.clip(new_sample, lb, ub)
            new_eval = limited_func(new_sample)
            if new_eval < func(best_sample):
                best_sample = new_sample
            
            if eval_count < remaining_budget // 2:
                dynamic_radius = 0.2 * (1.0 - eval_count / remaining_budget)
                samples = np.random.uniform(lb, ub, (3, self.dim))
                sample_evals = np.array([limited_func(np.clip(sample + dynamic_radius, lb, ub)) for sample in samples])
                best_idx = np.argmin(sample_evals)
                if sample_evals[best_idx] < func(best_sample):
                    best_sample = samples[best_idx]
        
        result = minimize(limited_func, best_sample, method='L-BFGS-B', bounds=list(zip(lb, ub)))
        
        return result.x