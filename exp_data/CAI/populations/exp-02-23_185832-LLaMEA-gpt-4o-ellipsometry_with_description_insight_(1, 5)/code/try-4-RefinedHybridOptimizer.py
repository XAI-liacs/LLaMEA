import numpy as np
from scipy.optimize import minimize

class RefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Extract bounds from the function
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Adaptive number of initial random samples based on remaining budget
        initial_samples = min(self.budget // 3, 5 * self.dim)
        
        # Randomly sample initial points within the bounds
        samples = np.random.uniform(lb, ub, (initial_samples, self.dim))
        sample_evals = np.array([func(sample) for sample in samples])

        # Reduce budget according to samples taken
        self.budget -= initial_samples

        # Find best initial sample
        best_idx = np.argmin(sample_evals)
        best_sample = samples[best_idx]
        best_val = sample_evals[best_idx]

        # Adaptive local optimization function
        def local_optimization(x0):
            nonlocal best_val, best_sample
            result = minimize(func, x0, method='nelder-mead', bounds=list(zip(lb, ub)),
                              options={'maxfev': self.budget, 'xatol': 1e-8, 'fatol': 1e-8})
            return result.x, result.fun

        # Perform local optimization from the best found sample
        while self.budget > 0:
            remaining_budget = self.budget
            final_sample, final_val = local_optimization(best_sample)
            if final_val < best_val:
                best_sample, best_val = final_sample, final_val

            # Reduce budget by the number of function evaluations used
            self.budget -= (remaining_budget - self.budget)

        # Return the best found solution and its function value
        return best_sample, best_val