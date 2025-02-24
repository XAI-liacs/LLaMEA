import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from skopt import gp_minimize

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
        
        # Step 2: Select the best initial guess and refine with Bayesian optimization
        best_indices = np.argsort(initial_values)[:3]
        best_guess = np.mean(initial_samples[best_indices], axis=0)  
        
        local_bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Replace local optimizer with Gaussian Process-based Bayesian optimization
        result = gp_minimize(
            func=lambda x: func(np.array(x)),
            dimensions=[(lb, ub) for lb, ub in bounds],
            n_calls=self.budget - self.evals,
            x0=best_guess.tolist(),
            random_state=42
        )
        
        self.evals += len(result.func_vals)
        
        while self.evals < self.budget:
            scale_factor = max(0.05, 0.1 * (1 - self.evals / self.budget))
            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb)), min(ub, result.x[i] + scale_factor * (ub - lb)))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            result = gp_minimize(
                func=lambda x: func(np.array(x)),
                dimensions=narrowed_bounds,
                n_calls=self.budget - self.evals,
                x0=result.x.tolist(),
                random_state=42
            )
            
            self.evals += len(result.func_vals)
        
        return np.array(result.x)