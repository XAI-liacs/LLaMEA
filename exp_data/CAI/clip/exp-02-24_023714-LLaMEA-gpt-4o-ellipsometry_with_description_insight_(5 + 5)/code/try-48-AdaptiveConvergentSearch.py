import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class AdaptiveConvergentSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluation_count = 0

    def __call__(self, func):
        # Initial Sobol sequence sampling for better coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        sample_points = sampler.random_base2(m=4)  # 16 points
        initial_guesses = qmc.scale(sample_points, func.bounds.lb, func.bounds.ub).tolist()
        best_solution = None
        best_value = float('inf')
        no_improvement_count = 0  # Track number of iterations without improvement

        # Define a wrapping function to count evaluations
        def wrapped_func(x):
            if self.evaluation_count >= self.budget:
                return float('inf')
            self.evaluation_count += 1
            return func(x)

        # Begin with a local optimizer
        for guess in initial_guesses:
            # Dynamically select the optimization method
            method = 'L-BFGS-B' if self.evaluation_count < self.budget / 2 else 'Nelder-Mead'
            result = minimize(wrapped_func, guess, method=method, bounds=list(zip(func.bounds.lb, func.bounds.ub)))

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                no_improvement_count = 0  # Reset counter if improvement is found
            else:
                no_improvement_count += 1  # Increment if no improvement

            # Update bounds based on the current best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))

            # If budget is exhausted or no improvement in 5 iterations, terminate
            if self.evaluation_count >= self.budget or no_improvement_count >= 5:
                break

        return best_solution