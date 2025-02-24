import numpy as np
from scipy.optimize import minimize

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define bounds from the function's bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Step 1: Uniform Sampling for Good Initial Guesses
        initial_guesses = np.random.uniform(lb, ub, size=(self.dim, self.dim))
        
        # Store the best solution found
        best_solution = None
        best_objective = float('inf')
        
        # Function to track remaining evaluations
        def track_evaluations(x):
            if self.evaluations < self.budget:
                self.evaluations += 1
                return func(x)
            else:
                raise Exception("Exceeded budget of function evaluations")

        # Define a function to adjust trust-region size based on current progress
        def adjust_trust_region(current_best, new_solution):
            delta = np.linalg.norm(current_best - new_solution)
            return max(0.1, min(0.5, delta))

        # Step 2: Local Optimization from Each Initial Guess
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break

            # Use L-BFGS-B for local search with initial trust region size
            trust_region_size = 0.5
            result = minimize(track_evaluations, guess, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'step_size': trust_region_size})
            
            # Update best solution if a new one is found
            if result.fun < best_objective:
                best_solution = result.x
                best_objective = result.fun
            
            # Adjust trust-region size dynamically
            if best_solution is not None:
                trust_region_size = adjust_trust_region(guess, best_solution)

        # If the budget allows, refine the best found solution with a final local search
        if self.evaluations < self.budget:
            result = minimize(track_evaluations, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'step_size': trust_region_size})
            best_solution = result.x
            best_objective = result.fun

        return best_solution