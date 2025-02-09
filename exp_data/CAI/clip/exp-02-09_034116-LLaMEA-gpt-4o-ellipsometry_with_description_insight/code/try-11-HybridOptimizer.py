import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol, LatinHypercube

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Exploration phase: Adaptive sampling strategy
        gradient_check = np.random.rand() > 0.5  # Randomly decide sampling strategy
        sampler = Sobol(d=self.dim, scramble=True) if gradient_check else LatinHypercube(d=self.dim)
        num_samples = max(1, self.budget // 10)
        samples = sampler.random_base2(m=int(np.log2(num_samples))) if gradient_check else sampler.random(num_samples)
        
        for sample in samples:
            sample_scaled = lb + sample * (ub - lb)
            value = func(sample_scaled)
            if value < best_value:
                best_value = value
                best_solution = sample_scaled
        
        # Exploitation phase: Local optimization using L-BFGS-B
        def wrapped_func(x):
            return func(x)
        
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxiter': self.budget - num_samples})
        
        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x
        
        return best_solution