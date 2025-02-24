import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class NovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define bounds from the function's bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Step 1: Improved Sampling for Good Initial Guesses using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=self.dim)
        initial_guesses = qmc.scale(sample, lb, ub)

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

        # Step 2: Local Optimization from Each Initial Guess
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break

            # Use L-BFGS-B for local search
            result = minimize(track_evaluations, guess, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            
            # Update best solution if a new one is found
            if result.fun < best_objective:
                best_solution = result.x
                best_objective = result.fun
        
        # If the budget allows, refine the best found solution with a final local search
        if self.evaluations < self.budget:
            result = minimize(track_evaluations, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            best_solution = result.x
            best_objective = result.fun

        return best_solution