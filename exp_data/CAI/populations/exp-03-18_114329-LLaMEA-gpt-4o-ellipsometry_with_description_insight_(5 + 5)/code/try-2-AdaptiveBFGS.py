import numpy as np
from scipy.optimize import minimize

class AdaptiveBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize counters and results
        remaining_budget = self.budget
        best_solution = None
        best_value = float('inf')

        # Uniformly sample the initial guesses in the parameter space
        num_initial_samples = min(5, self.budget // 10)
        initial_samples = [np.random.uniform(func.bounds.lb, func.bounds.ub) 
                           for _ in range(num_initial_samples)]
        
        # Define the objective function wrapper to track budget
        def objective_wrapper(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                raise RuntimeError("Budget exceeded")
            remaining_budget -= 1
            return func(x)

        # Run local optimization from each initial guess
        for initial_sample in initial_samples:
            result = minimize(
                objective_wrapper,
                initial_sample,
                method='L-BFGS-B',
                bounds=list(zip(func.bounds.lb, func.bounds.ub))
            )
            
            # Update best solution if found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Break if budget is near exhausted
            if remaining_budget < num_initial_samples:
                break

        return best_solution