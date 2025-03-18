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
        
        num_samples = min(max(10, int(self.budget * 0.3)), self.budget // 2)  # Adjusted sample size
        sampler = Sobol(d=self.dim, scramble=True)
        initial_samples = sampler.random(num_samples) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
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
            if self.evals < 0.8 * self.budget:
                scale_factor = max(0.05, 0.2 * (1 - self.evals / self.budget))  # Increased scale
            else:
                scale_factor = 0.05

            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb)), min(ub, result.x[i] + scale_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            result = minimize(
                func,
                best_guess,
                method='L-BFGS-B',
                bounds=narrowed_bounds,
                options={'maxfun': self.budget - self.evals}
            )
            
            self.evals += result.nfev
        
        return result.x