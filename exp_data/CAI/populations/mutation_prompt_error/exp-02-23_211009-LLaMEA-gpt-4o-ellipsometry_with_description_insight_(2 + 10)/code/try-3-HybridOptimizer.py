import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Get bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Calculate initial sampling points based on budget and dimension
        initial_samples = min(self.budget // 10, 100)
        remaining_budget = self.budget - initial_samples
        
        # Uniformly sample initial points using Sobol sequence
        sobol_sampler = Sobol(d=self.dim)
        samples = sobol_sampler.random_base2(m=int(np.log2(initial_samples)))
        samples = lb + samples * (ub - lb)
        best_value = float('inf')
        best_solution = None
        
        # Evaluate sampled points
        evaluations = 0
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Use local optimization with BFGS starting from the best sampled point
        def bounded_func(x):
            # Ensure the search does not go out of bounds
            return func(np.clip(x, lb, ub))
        
        # Use up remaining budget in local optimization
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        
        return result.x