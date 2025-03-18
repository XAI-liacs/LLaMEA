import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        num_samples = min(max(5, self.budget // 4), 12)

        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_vals = [func(sample) for sample in samples]

        self.budget -= num_samples
        
        best_idx = np.argmin(sample_vals)
        best_sample = samples[best_idx]

        lb = np.maximum(lb, best_sample - (ub - lb) * 0.15)
        ub = np.minimum(ub, best_sample + (ub - lb) * 0.15)
        
        velocity = np.zeros(self.dim)
        
        # Use adaptive learning rate
        learning_rate = 0.1 * (self.budget / (self.budget + 1))
        
        def budgeted_func(x):
            if self.budget <= 0:
                raise RuntimeError("Budget exceeded")
            self.budget -= 1
            return func(x)
        
        result = minimize(budgeted_func, best_sample, method='BFGS', bounds=list(zip(lb, ub)), 
                          options={'gtol': 1e-5, 'disp': False}, jac=None)

        if result.success:
            velocity = learning_rate * velocity + (1 - learning_rate) * result.x
            result.x += velocity
        
        # Restart mechanism if convergence stalls
        if not result.success:
            best_sample = samples[(best_idx + 1) % num_samples]  # Restart from next best

        return result.x if result.success else best_sample