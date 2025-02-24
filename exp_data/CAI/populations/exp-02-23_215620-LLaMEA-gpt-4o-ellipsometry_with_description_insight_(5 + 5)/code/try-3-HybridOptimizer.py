import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0
    
    def __call__(self, func):
        # Initial uniform sampling
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_guess = lb + (ub - lb) * np.random.rand(self.dim)
        
        # Initial function evaluation budget
        initial_budget = int(self.budget * 0.2)
        
        # Initial sampling
        best_guess = initial_guess
        best_value = func(best_guess)
        self.evals += 1
        
        for _ in range(initial_budget - 1):  # Utilize part of the budget for initial sampling
            if self.evals >= self.budget:
                break
            guess = lb + (ub - lb) * np.random.rand(self.dim)
            value = func(guess)
            self.evals += 1
            if value < best_value:
                best_value = value
                best_guess = guess
        
        # Use BFGS for local optimization
        options = {'maxiter': self.budget - self.evals}
        result = minimize(func, best_guess, method='BFGS', bounds=list(zip(lb, ub)), options=options)
        
        return result.x if result.success else best_guess