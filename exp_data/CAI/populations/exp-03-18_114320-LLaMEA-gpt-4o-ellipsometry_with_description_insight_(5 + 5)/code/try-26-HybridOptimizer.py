import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        sample_size = min(self.budget // 4, 15)  # Adjusted initial sampling size
        samples = np.random.uniform(lb, ub, (sample_size, self.dim))
        evaluations = [func(sample) for sample in samples]
        self.budget -= sample_size
        
        sorted_indices = np.argsort(evaluations)  # Identify top samples
        best_samples = samples[sorted_indices[:3]]  # Select top 3 samples
        
        best_eval = float('inf')
        best_solution = None

        for sample in best_samples:
            def local_optimization(x):
                return func(x)
            
            opts = {'maxiter': self.budget // 3, 'disp': False, 'learning_rate': 0.01}  # Added learning_rate
            result = minimize(local_optimization, sample, method='L-BFGS-B', bounds=list(zip(lb, ub)), options=opts)
            if result.fun < best_eval:
                best_eval = result.fun
                best_solution = result.x
        
        return best_solution