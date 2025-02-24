import numpy as np
from scipy.optimize import minimize

class ProgressiveSamplingHybridOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize best solution
        best_x = None
        best_fval = np.inf
        
        # Calculate the number of initial guesses based on a progressive sampling strategy
        initial_guesses = 5  # Start with a small number of initial guesses
        additional_guesses = max(0, self.budget // 10 - initial_guesses)
        
        # Use uniform sampling for initial guesses
        for _ in range(initial_guesses):
            x0 = lb + (ub - lb) * np.random.rand(self.dim)
            
            # Local optimization using Nelder-Mead
            result = minimize(func, x0, method='Nelder-Mead', bounds=list(zip(lb, ub)))
            
            # Update best solution found
            if result.fun < best_fval:
                best_fval = result.fun
                best_x = result.x
                
            # Decrement budget
            self.budget -= result.nfev
            if self.budget <= 0:
                break
        
        # Iteratively add more guesses if budget allows, focusing on areas around the best found solution
        for _ in range(additional_guesses):
            if self.budget <= 0:
                break
            # Create new guesses around the current best solution with slight perturbations
            perturbation = (ub - lb) * 0.1 * np.random.randn(self.dim)
            x0 = np.clip(best_x + perturbation, lb, ub)
            
            # Local optimization with the perturbed guess
            result = minimize(func, x0, method='Nelder-Mead', bounds=list(zip(lb, ub)))
            
            # Update best solution found
            if result.fun < best_fval:
                best_fval = result.fun
                best_x = result.x
                
            # Decrement budget
            self.budget -= result.nfev

        return best_x, best_fval