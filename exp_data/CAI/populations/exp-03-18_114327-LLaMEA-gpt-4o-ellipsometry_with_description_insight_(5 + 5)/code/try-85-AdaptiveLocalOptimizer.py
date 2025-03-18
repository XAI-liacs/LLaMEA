import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
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
        
        options = {'maxiter': self.budget - self.evaluations}
        result = minimize(wrapper_function, initial_guess, method='L-BFGS-B', bounds=np.stack((lb, ub), axis=1), options=options)

        while self.evaluations < self.budget:
            if best_value < 1e-6:  # Early stopping if a sufficiently good solution is found
                break
            std_dev_factor = np.std([best_solution]) * (self.budget - self.evaluations) / self.budget
            refined_bounds = [(max(lb[i], best_solution[i] - std_dev_factor * (ub[i] - lb[i])), 
                               min(ub[i], best_solution[i] + std_dev_factor * (ub[i] - lb[i]))) for i in range(self.dim)]
            refined_initial = np.random.uniform([b[0] for b in refined_bounds], [b[1] for b in refined_bounds])
            result = minimize(wrapper_function, refined_initial, method='L-BFGS-B', bounds=refined_bounds, options=options)

        return best_solution