import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Determine bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        remaining_budget = self.budget

        # Adaptive sampling for improved candidate diversity
        num_samples = min(max(5, self.budget // 20), remaining_budget // 2)
        initial_guesses = np.random.uniform(lb, ub, size=(num_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')

        # Evaluate initial guesses
        evaluations = 0
        for guess in initial_guesses:
            if evaluations >= remaining_budget:
                break
            value = func(guess)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Multi-start BFGS for robust local optimization
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= remaining_budget:
                return float('inf')
            value = func(x)
            evaluations += 1
            return value
        
        start_points = [best_solution] + list(initial_guesses[:min(3, len(initial_guesses))])
        for start_point in start_points:
            result = minimize(wrapped_func, start_point, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        return best_solution