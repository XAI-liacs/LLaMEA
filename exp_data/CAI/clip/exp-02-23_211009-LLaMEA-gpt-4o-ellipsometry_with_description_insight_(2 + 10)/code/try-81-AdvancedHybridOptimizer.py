import numpy as np
from scipy.optimize import minimize

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Stratified sampling to improve initial diversity
        strata = max(min(self.budget // (4 * self.dim), 50), 5)
        initial_samples = strata * 2
        remaining_budget = self.budget - initial_samples
        
        # Create stratified initial points
        samples = np.vstack([np.random.uniform(lb + strata*(ub-lb)/initial_samples, 
                                               lb + (strata+1)*(ub-lb)/initial_samples, 
                                               (strata, self.dim)) 
                             for strata in range(initial_samples // 2)])
        
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
        
        # Define a bounded function to ensure the search remains within the specified bounds
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        # Adaptive L-BFGS-B with gradient scaling for iterative refinement
        options = {'maxiter': remaining_budget, 'disp': False, 'gtol': 1e-8}
        result = minimize(bounded_func, best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)
        
        return result.x