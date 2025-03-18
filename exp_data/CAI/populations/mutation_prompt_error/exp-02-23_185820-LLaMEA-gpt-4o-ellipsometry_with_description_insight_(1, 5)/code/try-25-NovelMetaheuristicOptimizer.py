import numpy as np
from scipy.optimize import minimize

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        
        num_samples = min(max(7, self.budget // 3), self.budget // 2)
        
        # Step 1: Uniform sampling for initial exploration
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_samples, self.dim))
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        # Step 2: Select the best initial guess and refine with local optimizer
        best_idx = np.argmin(initial_values)
        best_guess = initial_samples[best_idx]
        
        local_bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Local optimizer using L-BFGS-B
        result = minimize(
            func,
            best_guess,
            method='L-BFGS-B',
            bounds=local_bounds,
            options={'maxfun': self.budget - self.evals}
        )
        
        self.evals += result.nfev

        # If evaluations are not yet exhausted, perform adaptive bound adjustment
        while self.evals < self.budget:
            scale_factor = max(0.05, 0.1 * (1 - self.evals / self.budget))
            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb)), min(ub, result.x[i] + scale_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            # Introduce dynamic adjustment of initial guess diversity
            diversity_adjustment = np.random.normal(0, 0.01, self.dim)
            dynamic_guess = result.x + diversity_adjustment
            
            # Another local search with adjusted bounds
            result = minimize(
                func,
                dynamic_guess,
                method='L-BFGS-B',
                bounds=narrowed_bounds,
                options={'maxfun': self.budget - self.evals}
            )
            
            self.evals += result.nfev
        
        return result.x