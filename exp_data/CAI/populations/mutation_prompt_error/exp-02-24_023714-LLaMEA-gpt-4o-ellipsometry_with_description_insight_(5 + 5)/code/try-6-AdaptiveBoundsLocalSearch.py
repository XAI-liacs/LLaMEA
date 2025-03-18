import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundsLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize budget counter
        self.evaluations = 0

        # Extract bounds from the provided function
        lb, ub = func.bounds.lb, func.bounds.ub

        # Uniformly sample initial guess within bounds
        initial_guess = np.random.uniform(lb, ub, self.dim)

        # Define the wrapper to count function evaluations
        def func_wrapper(x):
            if self.evaluations < self.budget:
                self.evaluations += 1
                return func(x)
            else:
                raise Exception("Budget exceeded")
        
        # Iteratively adjust bounds and perform local optimization
        while self.evaluations < self.budget:
            # Perform local optimization using BFGS
            result = minimize(func_wrapper, initial_guess, method='BFGS', bounds=list(zip(lb, ub)))

            # Update the initial guess
            initial_guess = result.x

            # Adaptively reduce bounds for further exploration
            lb = np.maximum(lb, initial_guess - (ub - lb) * 0.1)
            ub = np.minimum(ub, initial_guess + (ub - lb) * 0.1)

            # Break if optimization is successful
            if result.success or self.evaluations >= self.budget:  # Changed line
                break

        return initial_guess, result.fun