import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalBFGS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize best solution and its value
        best_solution = None
        best_value = float('inf')
        evals = 0
        
        # Initial uniform sampling for better starting point
        num_initial_samples = min(10, self.budget // 10)
        for _ in range(num_initial_samples):
            initial_guess = np.random.uniform(func.bounds.lb + 0.2 * (func.bounds.ub - func.bounds.lb), func.bounds.ub - 0.2 * (func.bounds.ub - func.bounds.lb))
            initial_value = func(initial_guess)
            evals += 1
            if initial_value < best_value:
                best_value = initial_value
                best_solution = initial_guess
        
        # Define the objective function for minimization
        def objective(x):
            nonlocal evals
            if evals >= self.budget:
                return float('inf')
            evals += 1
            return func(x)

        # Local optimization using BFGS with dynamic bounds adjustment
        bounds = [(low, high) for low, high in zip(func.bounds.lb, func.bounds.ub)]
        while evals < self.budget:
            result = minimize(objective, x0=best_solution, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                # Adjust the bounds based on current best solution
                bounds = [(max(low, x - (high - low) * 0.1), min(high, x + (high - low) * 0.1))
                          for (low, high), x in zip(bounds, best_solution)]
        
        return best_solution