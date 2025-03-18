import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Exploration phase: Sobol sequence sampling with dynamic adjustment
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        num_samples = max(1, int(np.sqrt(self.budget)))  # Use square root to dynamically adjust sample size
        samples = sobol_sampler.random(num_samples)
        top_k = min(len(samples), max(1, int(self.budget * 0.1)))  # Increased top-k percentage for better exploration

        # Evaluate and sort samples
        evaluated_samples = [(s, func(lb + s * (ub - lb))) for s in samples]
        evaluated_samples.sort(key=lambda x: x[1])
        
        top_samples = [s[0] for s in evaluated_samples[:top_k]]

        # Exploitation phase: Local optimization using L-BFGS-B from multiple samples
        allocated_budget = (self.budget - num_samples) // top_k
        for sample in top_samples:
            sample_scaled = lb + sample * (ub - lb)
            def wrapped_func(x):
                return func(x)
            
            result = minimize(wrapped_func, sample_scaled, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxiter': allocated_budget})
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution