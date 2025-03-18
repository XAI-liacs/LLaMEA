import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Adaptive number of initial samples
        num_samples = min(max(5, self.budget // 4), 12)  # Adjust sampling strategy

        # Uniform sampling to initialize
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_vals = [func(sample) for sample in samples]

        # Check budget usage
        self.budget -= num_samples
        
        # Find the best initial solution
        best_idx = np.argmin(sample_vals)
        best_sample = samples[best_idx]

        # Tighten bounds based on initial samples
        lb = np.maximum(lb, best_sample - (ub - lb) * 0.15)
        ub = np.minimum(ub, best_sample + (ub - lb) * 0.15)
        
        # Momentum-based update
        velocity = np.zeros(self.dim)
        
        # Define a function wrapper to track budget
        def budgeted_func(x):
            if self.budget <= 0:
                raise RuntimeError("Budget exceeded")
            self.budget -= 1
            return func(x)
        
        # Use BFGS for local optimization with adaptive momentum
        result = minimize(budgeted_func, best_sample, method='BFGS', bounds=list(zip(lb, ub)), 
                          options={'gtol': 1e-5, 'disp': False}, jac=None)

        # Incorporate momentum in updating sample
        if result.success:
            scaling_factor = 0.05 if self.budget > self.budget // 2 else 0.1  # Dynamic scaling factor
            velocity = 0.95 * velocity + scaling_factor * result.x  # Adjusted momentum coefficient
            result.x += velocity
        
        return result.x if result.success else best_sample