import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_samples = min(10, self.budget // 2)
        
        # Step 1: Uniform sampling for initial exploration
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_samples, self.dim))
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        # Step 2: Select the best initial guess and refine with local optimizer
        best_idx = np.argmin(initial_values)
        best_guess = initial_samples[best_idx]
        
        # Define the bounds again for the local optimizer
        local_bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Local optimizer using Nelder-Mead
        result = minimize(
            func,
            best_guess,
            method='Nelder-Mead',
            options={'maxfev': self.budget - self.evals}
        )
        
        # Update the number of evaluations used by the optimizer
        self.evals += result.nfev

        # Adaptive boundary exploration
        while self.evals < self.budget:
            # Adjust bounds progressively around the best solution
            exploration_factor = 0.1
            adjusted_bounds = [
                (max(lb, result.x[i] - exploration_factor * (ub - lb)), 
                 min(ub, result.x[i] + exploration_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            # Another local search with progressively adjusted bounds
            result = minimize(
                func,
                result.x,
                method='Nelder-Mead',
                bounds=adjusted_bounds,
                options={'maxfev': self.budget - self.evals}
            )
            
            # Update the number of evaluations
            self.evals += result.nfev
        
        return result.x