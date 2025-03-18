import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Extract bounds
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Define a small portion of the budget for exploration
        exploration_budget = int(0.2 * self.budget)
        exploitation_budget = self.budget - exploration_budget

        # Randomly sample initial guesses within bounds
        initial_guesses = np.random.uniform(lower_bounds, upper_bounds, (exploration_budget, self.dim))
        
        best_solution = None
        best_value = float('inf')

        # Exploration phase
        for guess in initial_guesses:
            value = func(guess)
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Adaptively adjust the bounds based on exploration results
        adaptive_lower_bounds = np.maximum(lower_bounds, best_solution - 0.1 * (upper_bounds - lower_bounds))
        adaptive_upper_bounds = np.minimum(upper_bounds, best_solution + 0.1 * (upper_bounds - lower_bounds))

        # Exploitation phase using BFGS
        result = minimize(func, best_solution, method='L-BFGS-B', bounds=list(zip(adaptive_lower_bounds, adaptive_upper_bounds)), options={'maxiter': exploitation_budget})

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        return best_solution