import numpy as np
from scipy.optimize import minimize

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        
        num_samples = min(max(7, int(self.budget * 0.4)), self.budget // 2)
        
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_samples, self.dim)) + np.random.normal(0, 0.01, (num_samples, self.dim))
        
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_samples
        
        best_idx = np.argmin(initial_values)
        best_guess = initial_samples[best_idx]
        
        local_bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        result = minimize(
            func,
            best_guess,
            method='L-BFGS-B',
            bounds=local_bounds,
            options={'maxfun': self.budget - self.evals}
        )
        
        self.evals += result.nfev

        while self.evals < self.budget:
            scale_factor = max(0.05, 0.15 * (1 - self.evals / self.budget))
            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb)), min(ub, result.x[i] + scale_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            # Adjusted line here to use the previous best parameters instead of best_guess
            result = minimize(
                func,
                result.x,  # Changed line
                method='L-BFGS-B',
                bounds=narrowed_bounds,
                options={'maxfun': self.budget - self.evals}
            )
            
            self.evals += result.nfev
        
        return result.x