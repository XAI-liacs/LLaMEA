import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class EnhancedNovelMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define bounds from the function's bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        current_bounds = np.array([lb, ub])
        
        # Step 1: Sobol Sequence for Initial Sampling
        sampler = Sobol(d=self.dim, scramble=True)
        initial_guesses = sampler.random_base2(m=int(np.log2(self.dim)))
        initial_guesses = lb + (ub - lb) * initial_guesses

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

            # Use BFGS for local search
            result = minimize(track_evaluations, guess, method='BFGS', bounds=list(zip(lb, ub)))
            
            # Update best solution if a new one is found
            if result.fun < best_objective:
                best_solution = result.x
                best_objective = result.fun
                # Update bounds around the best solution for refinement
                current_bounds = np.array([
                    np.maximum(lb, best_solution - 0.1 * (ub - lb)),
                    np.minimum(ub, best_solution + 0.1 * (ub - lb))
                ])

        # If the budget allows, refine the best found solution with a final local search
        if self.evaluations < self.budget:
            result = minimize(track_evaluations, best_solution, method='BFGS', bounds=list(zip(*current_bounds)))
            best_solution = result.x
            best_objective = result.fun

        return best_solution