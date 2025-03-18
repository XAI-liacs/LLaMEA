import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Adaptive number of initial samples
        num_samples = min(max(5, self.budget // 3), 15)  # Adjusted sampling strategy

        # Use Sobol sequence for more uniform initial sampling
        sobol_sampler = qmc.Sobol(d=self.dim, seed=42)
        samples = qmc.scale(sobol_sampler.random(num_samples), lb, ub)
        sample_vals = [func(sample) for sample in samples]

        # Check budget usage
        self.budget -= num_samples
        
        # Find the best initial solution
        best_idx = np.argmin(sample_vals)
        best_sample = samples[best_idx]
        
        # Define a function wrapper to track budget
        def budgeted_func(x):
            if self.budget <= 0:
                raise RuntimeError("Budget exceeded")
            self.budget -= 1
            return func(x)
        
        # Use BFGS for local optimization with early stopping
        result = minimize(budgeted_func, best_sample, method='BFGS', bounds=list(zip(lb, ub)), options={'gtol': 1e-5})

        return result.x if result.success else best_sample