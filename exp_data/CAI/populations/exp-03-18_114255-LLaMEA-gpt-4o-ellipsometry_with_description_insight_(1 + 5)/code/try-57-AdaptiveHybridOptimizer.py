import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guess = self.uniform_sample(bounds)
        result = self.local_search(func, initial_guess, bounds)
        
        while self.calls < self.budget:
            exploration_factor = np.exp(-self.calls / self.budget)  # New exploration strategy
            refined_bounds = self.adjust_bounds(result.x, bounds, shrink_factor=exploration_factor * 0.3)
            initial_guess = self.uniform_sample(refined_bounds)
            new_result = self.local_search(func, initial_guess, refined_bounds)
            
            if new_result.fun < result.fun:
                result = new_result
            else:
                # Implementing mutation strategy to escape local minima
                initial_guess = self.mutate_guess(result.x, refined_bounds, mutation_rate=0.1 * (1 - exploration_factor))
                result = self.local_search(func, initial_guess, refined_bounds)
        
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
    
    def mutate_guess(self, guess, bounds, mutation_rate):
        mutation = np.random.uniform(-mutation_rate, mutation_rate, self.dim)
        mutated_guess = guess + mutation
        # Ensure mutated guess stays within bounds
        return np.clip(mutated_guess, bounds[0], bounds[1])
    
    def adjust_bounds(self, best_guess, bounds, shrink_factor=0.3):
        new_bounds = np.zeros_like(bounds)
        for d in range(self.dim):
            center = best_guess[d]
            width = bounds[1, d] - bounds[0, d]
            new_bounds[0, d] = max(bounds[0, d], center - width * shrink_factor / 2)
            new_bounds[1, d] = min(bounds[1, d], center + width * shrink_factor / 2)
        return new_bounds