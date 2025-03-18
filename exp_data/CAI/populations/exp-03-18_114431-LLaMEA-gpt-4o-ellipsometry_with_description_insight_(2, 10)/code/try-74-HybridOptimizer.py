import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

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

        # Sobol sequence sampling to initialize
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random_base2(m=int(np.ceil(np.log2(num_samples))))
        samples = lb + samples * (ub - lb)
        sample_vals = [func(sample) for sample in samples]

        # Check budget usage
        self.budget -= num_samples
        
        # Find the best initial solution
        best_idx = np.argmin(sample_vals)
        best_sample = samples[best_idx]

        # Dynamic tightening of bounds based on variance
        sample_variance = np.var(samples, axis=0)
        lb = np.maximum(lb, best_sample - sample_variance * 0.15)
        ub = np.minimum(ub, best_sample + sample_variance * 0.15)
        
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
            adaptive_momentum = 0.95 - (0.05 * (self.budget / 100))  
            velocity = adaptive_momentum * velocity + 0.05 * result.x
            result.x += velocity
        
        return result.x if result.success else best_sample