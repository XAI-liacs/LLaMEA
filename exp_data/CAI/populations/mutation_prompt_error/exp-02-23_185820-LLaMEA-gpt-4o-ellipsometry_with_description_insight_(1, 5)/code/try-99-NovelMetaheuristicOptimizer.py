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
        
        # Step 2: Select the best initial guesses and refine with local optimizer
        best_indices = np.argsort(initial_values)[:3]
        best_guess = np.mean(initial_samples[best_indices], axis=0)
        
        # Define the bounds for the local optimizer
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

        # Step 3: Use simulated annealing-like adaptive bound adjustment
        while self.evals < self.budget:
            scale_factor = max(0.05, 0.1 * (1 - self.evals / self.budget))
            temperature = 1.0 - (self.evals / self.budget)
            narrowed_bounds = [
                (max(lb, result.x[i] - scale_factor * (ub - lb) * temperature), 
                 min(ub, result.x[i] + scale_factor * (ub - lb) * temperature))
                for i, (lb, ub) in enumerate(bounds)
            ]
            
            perturbation = temperature * np.random.uniform(-1, 1, self.dim) * (bounds[:, 1] - bounds[:, 0]) * scale_factor
            perturbed_initial = np.clip(result.x + perturbation, func.bounds.lb, func.bounds.ub)
            
            result = minimize(
                func,
                perturbed_initial,
                method='L-BFGS-B',
                bounds=narrowed_bounds,
                options={'maxfun': self.budget - self.evals}
            )
            
            self.evals += result.nfev
        
        return result.x