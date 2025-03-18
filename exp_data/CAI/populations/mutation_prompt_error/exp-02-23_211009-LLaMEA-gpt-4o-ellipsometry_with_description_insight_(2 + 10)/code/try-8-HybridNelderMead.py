import numpy as np
from scipy.optimize import minimize

class HybridNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        # Extract bounds for the problem
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Generate initial guesses by uniform sampling within bounds
        num_initial_points = min(5, self.budget // 2)  # number of initial points
        initial_guesses = np.random.uniform(lb, ub, size=(num_initial_points, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break
            
            # Use Nelder-Mead for local optimization with adaptive restart
            result = minimize(func, guess, method='Nelder-Mead', options={'initial_simplex': np.random.uniform(lb, ub, size=(self.dim+1, self.dim))})
            self.evaluations += result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution