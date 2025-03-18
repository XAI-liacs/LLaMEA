import numpy as np
from scipy.optimize import minimize

class AdaptiveBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize variables
        evaluations = 0
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        initial_guess = np.random.uniform(bounds[:, 0], bounds[:, 1])
        best_solution = initial_guess
        best_value = float('inf')

        # Define the local optimization function
        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')  # Ensure no more calls after budget
            evaluations += 1
            return func(x)

        # Perform iterative optimization with adaptive bounds
        while evaluations < self.budget:
            # Optimize using BFGS within current bounds
            result = minimize(local_optimization, initial_guess, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Adapt bounds towards the best solution found
            alpha = 0.1  # Bounds tightening rate
            bounds[:, 0] = np.maximum(bounds[:, 0], best_solution - alpha * (bounds[:, 1] - bounds[:, 0]))
            bounds[:, 1] = np.minimum(bounds[:, 1], best_solution + alpha * (bounds[:, 1] - bounds[:, 0]))
            initial_guess = np.random.uniform(bounds[:, 0], bounds[:, 1])

        return best_solution