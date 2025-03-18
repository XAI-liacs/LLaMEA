import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guess = self.sobol_sample(bounds)  # Changed initial sampling strategy
        result = self.local_search(func, initial_guess, bounds)
        
        while self.calls < self.budget:
            refined_bounds = self.adjust_bounds(result.x, bounds)
            initial_guess = self.uniform_sample(refined_bounds)
            new_result = self.local_search(func, initial_guess, refined_bounds)
            
            if new_result.fun < result.fun:
                result = new_result
        
        return result.x, result.fun
    
    def local_search(self, func, initial_guess, bounds):
        def wrapped_func(x):
            self.calls += 1
            return func(x)
        
        options = {'maxiter': min(self.budget - self.calls, 100)}
        result = minimize(wrapped_func, initial_guess, method='Nelder-Mead', options=options)
        result = minimize(wrapped_func, result.x, method='BFGS', options=options)

        return result
    
    def uniform_sample(self, bounds):
        return np.random.uniform(bounds[0], bounds[1], self.dim)
    
    def sobol_sample(self, bounds):  # New sampling function using Sobol sequence
        sampler = Sobol(d=self.dim, scramble=False)
        sample = sampler.random_base2(1)[0]  # Generate a single Sobol point
        scaled_sample = bounds[0] + sample * (bounds[1] - bounds[0])
        return scaled_sample
    
    def adjust_bounds(self, best_guess, bounds, shrink_factor=0.3):
        new_bounds = np.zeros_like(bounds)
        for d in range(self.dim):
            center = best_guess[d]
            width = bounds[1, d] - bounds[0, d]
            new_bounds[0, d] = max(bounds[0, d], center - width * shrink_factor / 2)
            new_bounds[1, d] = min(bounds[1, d], center + width * shrink_factor / 2)
        return new_bounds