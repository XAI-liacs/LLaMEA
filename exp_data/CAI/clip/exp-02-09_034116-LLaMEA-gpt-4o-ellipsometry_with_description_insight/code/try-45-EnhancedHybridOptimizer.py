import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Exploration phase: Adaptive Sobol sequence sampling
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        adaptive_samples = max(1, int(np.sqrt(self.budget)))  # Adaptive sample size based on square root of budget
        samples = sobol_sampler.random(adaptive_samples)
        top_k = min(len(samples), int(self.budget * 0.1))  # Adjusted top-k percentage
        evaluated_samples = [(lb + s * (ub - lb), func(lb + s * (ub - lb))) for s in samples]
        evaluated_samples.sort(key=lambda x: x[1])
        top_samples = [s[0] for s in evaluated_samples[:top_k]]
        
        # Exploitation phase: Multi-start local optimization using L-BFGS-B
        for sample in top_samples:
            def wrapped_func(x):
                return func(x)

            result = minimize(wrapped_func, sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxfun': (self.budget // top_k)})
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution