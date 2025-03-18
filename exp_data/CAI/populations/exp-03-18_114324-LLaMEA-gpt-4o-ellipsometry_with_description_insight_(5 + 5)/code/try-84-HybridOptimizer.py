import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub

        # Ensure bounds are valid
        if np.any(ub <= lb):
            raise ValueError("All upper bounds must be greater than lower bounds.")

        # Calculate the number of initial samples
        initial_sample_count = min(10, self.budget // 10)

        # Use Latin Hypercube Sampling for initial solutions within bounds
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=initial_sample_count)
        initial_guesses = qmc.scale(sample, lb, ub)
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        for guess in initial_guesses:
            # Use dual local optimizers in parallel starting from each initial guess
            res1 = minimize(func, guess, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxfun': (self.budget - evaluations) // 2})
            res2 = minimize(func, guess, method='Nelder-Mead', bounds=list(zip(lb, ub)), options={'maxfev': (self.budget - evaluations) // 2})
            
            evaluations += res1.nfev + res2.nfev
            
            if res1.fun < best_value:
                best_value = res1.fun
                best_solution = res1.x
            if res2.fun < best_value:
                best_value = res2.fun
                best_solution = res2.x
            
            if evaluations >= self.budget:
                break

        return best_solution, best_value