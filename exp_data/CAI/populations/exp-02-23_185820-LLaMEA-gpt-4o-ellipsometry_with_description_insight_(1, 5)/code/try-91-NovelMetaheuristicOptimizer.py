import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        
        num_samples = min(max(7, int(self.budget * 0.4)), self.budget // 2)
        
        # Step 1: Adaptive uniform sampling for initial exploration
        sampler = LatinHypercube(d=self.dim)
        initial_samples = sampler.random(n=num_samples) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        # Step 2: Select the best initial guess and refine with local optimizer
        best_indices = np.argsort(initial_values)[:3]
        best_guess = np.mean(initial_samples[best_indices], axis=0)
        
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
            scale_factor = max(0.05, 0.1 * (1 - self.evals / self.budget))  # Reduce factor incrementally
            temperature = 1.0 - (self.evals / self.budget)  # Simulated annealing temperature
            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb) * temperature), 
                 min(ub, result.x[i] + scale_factor * (ub - lb) * temperature))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            # Use result.x instead of best_guess to further leverage refined data
            result = minimize(
                func,
                result.x,  # Changed from best_guess to result.x
                method='Nelder-Mead',  # Changed from L-BFGS-B to Nelder-Mead
                bounds=narrowed_bounds,
                options={'maxfun': self.budget - self.evals}
            )
            
            # Update the number of evaluations
            self.evals += result.nfev
        
        return result.x