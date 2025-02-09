import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Exploration phase: Sobol sequence sampling
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        num_samples = max(1, int(np.log2(self.budget)))  # Dynamically adjust sample size
        samples = sobol_sampler.random_base2(m=int(np.log2(num_samples)))
        top_k = min(len(samples), int(self.budget * 0.1))  # Adjusted top-k percentage from 5% to 10%
        top_samples = sorted(samples[:top_k], key=lambda s: func(lb + s * (ub - lb)))
        
        # Exploitation phase: Local optimization using L-BFGS-B from multiple samples
        for sample in top_samples:
            sample_scaled = lb + sample * (ub - lb)
            def wrapped_func(x):
                return func(x)
            
            result = minimize(wrapped_func, sample_scaled, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxiter': (self.budget // top_k) - num_samples})
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution