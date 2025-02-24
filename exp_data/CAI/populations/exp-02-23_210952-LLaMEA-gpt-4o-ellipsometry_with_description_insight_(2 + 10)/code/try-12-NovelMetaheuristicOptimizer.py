import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract the bounds and prepare for optimizations
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        # Calculate the number of initial samples based on the available budget
        num_initial_samples = min(self.budget // 4, 8)  # Use a smaller portion for sampling
        remaining_budget = self.budget - num_initial_samples

        # Initialize the best solution found so far
        best_solution = None
        best_score = float('inf')

        # Step 1: Uniformly sample the initial solutions
        initial_solutions = np.random.uniform(lower_bounds, upper_bounds, (num_initial_samples, self.dim))
        
        # Step 2: Evaluate initial samples in parallel to improve efficiency
        with ThreadPoolExecutor() as executor:
            scores = list(executor.map(func, initial_solutions))
        
        # Identify the best initial solution
        for solution, score in zip(initial_solutions, scores):
            if score < best_score:
                best_score = score
                best_solution = solution
        
        # Step 3: Adaptive sampling to redistribute remaining budget
        adaptive_samples = min(remaining_budget // 2, 4)
        remaining_budget -= adaptive_samples
        adaptive_solutions = np.random.uniform(lower_bounds, upper_bounds, (adaptive_samples, self.dim))
        
        # Evaluate adaptive samples in parallel
        with ThreadPoolExecutor() as executor:
            adaptive_scores = list(executor.map(func, adaptive_solutions))
        
        # Update the best solution from adaptive sampling
        for solution, score in zip(adaptive_solutions, adaptive_scores):
            if score < best_score:
                best_score = score
                best_solution = solution

        # Step 4: Use BFGS local optimization from the best-found sample
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': remaining_budget, 'ftol': 1e-9})

        # Return the best found solution
        return result.x if result.success else best_solution