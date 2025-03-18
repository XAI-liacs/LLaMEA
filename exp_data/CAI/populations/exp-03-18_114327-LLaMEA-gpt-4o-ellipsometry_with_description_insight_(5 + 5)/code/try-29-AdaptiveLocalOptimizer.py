import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initial uniform sampling within bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_guess = np.random.uniform(lb, ub, size=(self.dim,))
        best_solution = None
        best_value = float('inf')
        
        def wrapper_function(x):
            nonlocal best_solution, best_value
            if self.evaluations < self.budget:
                value = func(x)
                self.evaluations += 1
                if value < best_value:
                    best_value = value
                    best_solution = x
                return value
            else:
                return float('inf')
        
        # Since the function is smooth and low-dimensional, use a local optimizer like BFGS
        options = {'maxiter': self.budget - self.evaluations}
        result = minimize(wrapper_function, initial_guess, method='L-BFGS-B', bounds=np.stack((lb, ub), axis=1), options=options)

        # If there is budget left, iteratively refine the bounds based on standard deviation of best solutions
        while self.evaluations < self.budget:
            std_dev_factor = np.std([best_solution]) * (self.budget - self.evaluations) / self.budget
            refined_bounds = [(max(lb[i], best_solution[i] - std_dev_factor * (ub[i] - lb[i])), 
                               min(ub[i], best_solution[i] + std_dev_factor * (ub[i] - lb[i]))) for i in range(self.dim)]
            refined_initial = np.random.uniform([b[0] for b in refined_bounds], [b[1] for b in refined_bounds])
            result = minimize(wrapper_function, refined_initial, method='L-BFGS-B', bounds=refined_bounds, options=options)

        return best_solution

# Example usage:
# optimizer = AdaptiveLocalOptimizer(budget=100, dim=2)
# best_parameters = optimizer(func)