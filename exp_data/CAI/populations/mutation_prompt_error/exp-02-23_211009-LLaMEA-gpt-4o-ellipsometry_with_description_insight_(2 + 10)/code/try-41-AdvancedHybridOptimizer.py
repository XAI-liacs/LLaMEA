import numpy as np
from scipy.optimize import minimize, differential_evolution

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Dynamically adjust initial sampling points based on budget and dimensionality
        initial_samples = max(min(self.budget // (3 * self.dim), 100), 10) 
        remaining_budget = self.budget - initial_samples
        
        # Use differential evolution for initial phase exploration
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        result = differential_evolution(func, bounds, maxiter=initial_samples, popsize=5, disp=False)  
        initial_best_solution = result.x
        evaluations = result.nfev
        
        # Define a bounded function to ensure the search remains within the specified bounds
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        
        # Use the remaining budget efficiently in local optimization with adaptive L-BFGS-B
        options = {'maxiter': remaining_budget - evaluations, 'disp': False}
        result = minimize(bounded_func, initial_best_solution, method='L-BFGS-B', bounds=np.array([lb, ub]).T, options=options)  
        
        return result.x