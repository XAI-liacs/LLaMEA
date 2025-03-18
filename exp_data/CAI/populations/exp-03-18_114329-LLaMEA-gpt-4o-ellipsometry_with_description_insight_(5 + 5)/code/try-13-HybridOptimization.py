import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Use Sobol sequence for initial sampling to enhance diversity.
        num_initial_guesses = min(20, self.budget // 2)
        best_solution = None
        best_value = float('inf')
        
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Generate initial guesses within the bounds using Sobol sequence
        sobol = qmc.Sobol(d=self.dim, scramble=False)
        initial_guesses = qmc.scale(sobol.random(num_initial_guesses), lb, ub)
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
                new_bounds = [(max(lb[i], res.x[i] - 0.1 * (ub[i] - lb[i])), 
                               min(ub[i], res.x[i] + 0.1 * (ub[i] - lb[i]))) for i in range(self.dim)]
                for _ in range(3):  # Retry with slightly adjusted bounds
                    if evaluations >= self.budget:
                        break
                    res = minimize(func, res.x, method='L-BFGS-B', bounds=new_bounds)
                    evaluations += res.nfev
                    if res.fun < best_value:
                        best_value = res.fun
                        best_solution = res.x
                        
        return best_solution