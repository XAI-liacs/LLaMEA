import numpy as np
from scipy.optimize import minimize

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        
        # Adjusted line for increased initial sample size
        num_samples = min(max(7, int(self.budget * 0.4)), self.budget // 2)
        
        # Step 1: Uniform sampling for initial exploration
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_samples, self.dim))
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        # Step 2: Select the best initial guess and refine with local optimizer
        best_idx = np.argmin(initial_values)
        best_guess = initial_samples[best_idx]
        
        # Define the bounds again for the local optimizer
        local_bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Local optimizer using L-BFGS-B
        result = minimize(
            func,
            best_guess,
            method='L-BFGS-B',
            bounds=local_bounds,
            options={'maxfun': self.budget - self.evals}
        )
        
        # Update the number of evaluations used by the optimizer
        self.evals += result.nfev

        # If evaluations are not yet exhausted, perform adaptive bound adjustment
        while self.evals < self.budget:
            # Refined dynamic scaling factor based on convergence rate
            scale_factor = max(0.05, 0.15 * (1 - self.evals / self.budget))  # Changed dynamic scaling factor
            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb)), min(ub, result.x[i] + scale_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            # Another local search with adjusted bounds
            result = minimize(
                func,
                result.x,
                method='L-BFGS-B',
                bounds=narrowed_bounds,
                options={'maxfun': self.budget - self.evals}
            )
            
            # Update the number of evaluations
            self.evals += result.nfev
        
        return result.x