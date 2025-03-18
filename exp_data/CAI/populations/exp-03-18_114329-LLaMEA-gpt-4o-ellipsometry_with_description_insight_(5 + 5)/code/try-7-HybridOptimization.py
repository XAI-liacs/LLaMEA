import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Uniform random sampling to generate initial guesses.
        num_initial_guesses = min(10, self.budget // 2)
        best_solution = None
        best_value = float('inf')
        
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Generate initial guesses within the bounds
        initial_guesses = np.random.uniform(lb, ub, (num_initial_guesses, self.dim))
        evaluations = 0

        for initial_guess in initial_guesses:
            if evaluations >= self.budget:
                break
            
            # Local optimization using BFGS
            res = minimize(func, initial_guess, method='L-BFGS-B', bounds=zip(lb, ub))
            evaluations += res.nfev

            # Update best solution found
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            # Adjust bounds and constraints iteratively
            if evaluations < self.budget:
                factor = 0.2  # Changed from 0.1 to 0.2 for deeper exploration
                new_bounds = [(max(lb[i], res.x[i] - factor * (ub[i] - lb[i])), 
                               min(ub[i], res.x[i] + factor * (ub[i] - lb[i]))) for i in range(self.dim)]
                for _ in range(3):  # Retry with slightly adjusted bounds
                    if evaluations >= self.budget:
                        break
                    res = minimize(func, res.x, method='L-BFGS-B', bounds=new_bounds)
                    evaluations += res.nfev
                    if res.fun < best_value:
                        best_value = res.fun
                        best_solution = res.x
                        
        return best_solution