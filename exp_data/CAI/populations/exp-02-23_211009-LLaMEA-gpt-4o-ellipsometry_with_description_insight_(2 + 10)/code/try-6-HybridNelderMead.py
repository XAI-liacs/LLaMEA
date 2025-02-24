import numpy as np
from scipy.optimize import minimize

class HybridNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Generate initial guesses using a more refined method
        num_initial_points = min(10, self.budget // 3)  # slightly increase initial points
        initial_guesses = np.random.uniform(lb, ub, size=(num_initial_points, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        values = []
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break
            value = func(guess)
            values.append((value, guess))
            self.evaluations += 1
        
        # Sort initial guesses based on function value and select top candidates
        values.sort()
        top_candidates = [x[1] for x in values[:num_initial_points//2]]  # keep top 50%
        
        for guess in top_candidates:
            if self.evaluations >= self.budget:
                break
            result = minimize(func, guess, method='Nelder-Mead', bounds=[(lb[i], ub[i]) for i in range(self.dim)])
            self.evaluations += result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution