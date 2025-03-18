import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = func.bounds
        lower_bound = bounds.lb
        upper_bound = bounds.ub

        # Adaptive bounds adjustment parameters
        adjustment_factor = 0.1
        convergence_threshold = 1e-5
        early_stopping_threshold = 1e-6 # Enhanced early stopping condition

        # Uniformly sample initial points
        num_initial_points = 5  # Multiple starting points for robustness
        initial_guesses = [lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound) 
                           for _ in range(num_initial_points)]
        
        best_solution = initial_guesses[0]
        best_value = func(best_solution)
        self.evaluations += 1
        
        for initial_guess in initial_guesses:
            # Local optimization using BFGS
            def local_optimize(x0):
                nonlocal best_solution, best_value
                result = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lower_bound, upper_bound)))
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x

            while self.evaluations < self.budget:
                # Perform local search
                local_optimize(best_solution)

                # Dynamically adjust bounds
                for i in range(self.dim):
                    lower_bound[i] = max(bounds.lb[i], best_solution[i] - adjustment_factor * (upper_bound[i] - lower_bound[i]))
                    upper_bound[i] = min(bounds.ub[i], best_solution[i] + adjustment_factor * (upper_bound[i] - lower_bound[i]))

                # Convergence check
                if np.linalg.norm(best_solution - initial_guess) < convergence_threshold:
                    break
                
                if best_value < early_stopping_threshold:  # Early stopping if close enough
                    break

                initial_guess = best_solution
                self.evaluations += 1

        return best_solution