import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class RefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Extract bounds from the function
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Number of initial samples using Sobol sequence for better coverage
        initial_samples = min(self.budget // 2, 10 * self.dim)
        
        # Generate Sobol sequence for initial sampling within the bounds
        sobol = Sobol(d=self.dim, scramble=True)
        samples = sobol.random_base2(m=int(np.log2(initial_samples)))
        samples = lb + (ub - lb) * samples
        sample_evals = np.array([func(sample) for sample in samples])

        # Reduce budget according to samples taken
        self.budget -= initial_samples

        # Find best initial sample
        best_idx = np.argmin(sample_evals)
        best_sample = samples[best_idx]
        best_val = sample_evals[best_idx]

        # Use BFGS for local optimization
        def local_optimization(x0):
            nonlocal best_val, best_sample
            result = minimize(func, x0, method='BFGS', bounds=list(zip(lb, ub)),
                              options={'maxiter': self.budget, 'gtol': 1e-8})
            return result.x, result.fun

        # Perform local optimization from the best found sample
        if self.budget > 0:
            final_sample, final_val = local_optimization(best_sample)
            if final_val < best_val:
                best_sample, best_val = final_sample, final_val
        
        # Return the best found solution and its function value
        return best_sample, best_val