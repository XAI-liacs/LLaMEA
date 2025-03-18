import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        sample_size = min(self.budget // 3, 20)  # Adjusted initial sampling size
        samples = np.random.uniform(lb, ub, (sample_size, self.dim))
        evaluations = [func(sample) for sample in samples]
        self.budget -= sample_size
        
        sorted_indices = np.argsort(evaluations)  
        top_fraction = 0.2  # Use top 20% of samples
        top_n = max(1, int(sample_size * top_fraction))
        best_samples = samples[sorted_indices[:top_n]]  # Select top samples
        
        best_eval = float('inf')
        best_solution = None

        for sample in best_samples:
            def local_optimization(x):
                return func(x)
            
            opts = {'maxiter': self.budget // top_n, 'disp': False}  # Adjusted maxiter
            result = minimize(local_optimization, sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options=opts)
            if result.fun < best_eval:
                best_eval = result.fun
                best_solution = result.x
        
        return best_solution