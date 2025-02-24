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
        
        # Number of initial random samples
        initial_samples = min(self.budget // 2, 10 * self.dim)

        # Randomly sample initial points within the bounds
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])

        # Reduce budget according to samples taken
        self.budget -= initial_samples

        # Find best initial sample
        best_idx = np.argmin(sample_evals)
        best_sample = samples[best_idx]
        best_val = sample_evals[best_idx]

        # Use Nelder-Mead for local optimization
        def local_optimization(x0):
            nonlocal best_val, best_sample
            result = minimize(func, x0, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-8, 'fatol': 1e-8})
            return result.x, result.fun

        # Perform local optimization from the best found sample
        if self.budget > 0:
            final_sample, final_val = local_optimization(best_sample)
            if final_val < best_val:
                best_sample, best_val = final_sample, final_val
        
        # Return the best found solution and its function value
        return best_sample, best_val