import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_history = []  # Store history of best found solutions
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guess = self.improved_initial_guess(func, bounds)
        result = self.local_search(func, initial_guess, bounds)
        self.best_history.append((result.x, result.fun))  # Track solutions

        while self.calls < self.budget:
            refined_bounds = self.adjust_bounds(result.x, bounds, shrink_factor=max(0.1, 0.3 * (1 - self.calls / self.budget)))
            initial_guess = self.dynamic_initial_guess(refined_bounds)  # Use dynamic initial guess
            new_result = self.local_search(func, initial_guess, refined_bounds)
            
            if new_result.fun < result.fun:
                result = new_result
                self.best_history.append((result.x, result.fun))
        
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
    
    def improved_initial_guess(self, func, bounds):
        # Use middle point in bounds and perturb it based on a small local gradient estimate
        midpoint = (bounds[0] + bounds[1]) / 2
        small_step = 1e-5
        grad_estimate = np.zeros(self.dim)
        for d in range(self.dim):
            step_vector = np.zeros(self.dim)
            step_vector[d] = small_step
            grad_estimate[d] = (func(midpoint + step_vector) - func(midpoint)) / small_step
        return midpoint - grad_estimate * small_step

    def dynamic_initial_guess(self, bounds):
        # Leverage historical best solutions to inform initial guess
        if self.best_history:
            historical_best = min(self.best_history, key=lambda x: x[1])[0]
            return self.uniform_sample(bounds) * 0.5 + historical_best * 0.5
        else:
            return self.uniform_sample(bounds)

    def adjust_bounds(self, best_guess, bounds, shrink_factor=0.3):
        new_bounds = np.zeros_like(bounds)
        for d in range(self.dim):
            center = best_guess[d]
            width = bounds[1, d] - bounds[0, d]
            new_bounds[0, d] = max(bounds[0, d], center - width * shrink_factor / 2)
            new_bounds[1, d] = min(bounds[1, d], center + width * shrink_factor / 2)
        return new_bounds