import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract the bounds and prepare for optimizations
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        # Calculate the number of initial samples based on the available budget
        num_initial_samples = max(self.budget // 4, 5)  # Adjusted for new strategy
        remaining_budget = self.budget - num_initial_samples

        # Initialize the best solution found so far
        best_solution = None
        best_score = float('inf')

        # Step 1: Simulated Annealing for initial exploration
        temperature = 1.0
        cooling_rate = 0.9
        initial_solution = np.random.uniform(lower_bounds, upper_bounds, self.dim)
        current_solution = initial_solution
        current_score = func(current_solution)

        for _ in range(num_initial_samples):
            if remaining_budget <= 0:
                break
            # Generate a new candidate solution
            candidate_solution = current_solution + np.random.normal(0, 0.1, self.dim) * (upper_bounds - lower_bounds)
            candidate_solution = np.clip(candidate_solution, lower_bounds, upper_bounds)
            
            # Evaluate the function
            candidate_score = func(candidate_solution)
            remaining_budget -= 1

            # Acceptance criterion (Metropolis criterion)
            if candidate_score < current_score or np.random.rand() < np.exp((current_score - candidate_score) / temperature):
                current_solution, current_score = candidate_solution, candidate_score
                if current_score < best_score:
                    best_solution, best_score = current_solution, current_score

            # Decrease temperature
            temperature *= cooling_rate

        # Step 2: Use BFGS local optimization from the best initial samples
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget, 'ftol': 1e-9})

        # Return the best found solution
        return result.x if result.success else best_solution