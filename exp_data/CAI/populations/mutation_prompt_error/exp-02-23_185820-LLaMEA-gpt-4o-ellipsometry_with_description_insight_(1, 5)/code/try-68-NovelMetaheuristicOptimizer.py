import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        
        num_samples = min(max(7, int(self.budget * 0.4)), self.budget // 2)
        
        # Step 1: Adaptive uniform sampling for initial exploration
        sampler = Sobol(d=self.dim, scramble=True)
        initial_samples = sampler.random_base2(m=int(np.log2(num_samples))) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        # Step 2: Select the best initial guess and refine with local optimizer
        best_idx = np.argmin(initial_values)
        second_best_idx = np.argmin([v for i, v in enumerate(initial_values) if i != best_idx])
        best_guess = (initial_samples[best_idx] + initial_samples[second_best_idx]) / 2  # Hybrid of top two samples
        
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
            scale_factor = max(0.05, 0.1 * (1 - self.evals / self.budget))  # Adjusted factor
            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb)), min(ub, result.x[i] + scale_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            # Use best_guess instead of result.x to further leverage initial sampling data
            result = minimize(
                func,
                best_guess,
                method='L-BFGS-B',
                bounds=narrowed_bounds,
                options={'maxfun': self.budget - self.evals}
            )
            
            # Update the number of evaluations
            self.evals += result.nfev
        
        # Golden-section refinement step
        result.x = result.x + 0.618 * (best_guess - result.x)
        
        return result.x