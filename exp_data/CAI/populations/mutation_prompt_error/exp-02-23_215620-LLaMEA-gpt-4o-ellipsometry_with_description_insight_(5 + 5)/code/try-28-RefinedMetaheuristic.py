import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class RefinedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define bounds from the function's bounds
        lb, ub = func.bounds.lb, func.bounds.ub

        # Step 1: Sobol Sequence Sampling for Better Coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sobol_samples = sampler.random_base2(m=int(np.log2(self.dim)))
        initial_guesses = qmc.scale(sobol_samples, lb, ub)

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

        # Step 2: Local Optimization Using L-BFGS-B for Bound Constraints
        for guess in initial_guesses:
            if self.evaluations >= self.budget:
                break

            # Use L-BFGS-B for local search
            result = minimize(track_evaluations, guess, method='L-BFGS-B', bounds=list(zip(lb, ub)))
            
            # Update best solution if a new one is found
            if result.fun < best_objective:
                best_solution = result.x
                best_objective = result.fun

        # Step 3: Dynamic Bounds Tightening
        if best_solution is not None:
            tightened_lb = np.maximum(lb, best_solution - 0.1 * (ub - lb))
            tightened_ub = np.minimum(ub, best_solution + 0.1 * (ub - lb))
            
            # Refine the best found solution with a final local search
            if self.evaluations < self.budget:
                result = minimize(track_evaluations, best_solution, method='L-BFGS-B', bounds=list(zip(tightened_lb, tightened_ub)))
                best_solution = result.x
                best_objective = result.fun

        return best_solution