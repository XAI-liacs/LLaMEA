import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class AdaptiveConvergentSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluation_count = 0
        self.initial_budget_fraction = 0.25

    def __call__(self, func):
        # Initial Sobol sequence sampling for better coverage
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        m = int(np.log2(self.budget * self.initial_budget_fraction))  # Determine m based on budget fraction
        sample_points = sampler.random_base2(m=m)
        initial_guesses = qmc.scale(sample_points, func.bounds.lb, func.bounds.ub).tolist()
        best_solution = None
        best_value = float('inf')

        # Define a wrapping function to count evaluations
        def wrapped_func(x):
            if self.evaluation_count >= self.budget:
                return float('inf')
            self.evaluation_count += 1
            return func(x)

        # Start with local optimization for each initial guess
        for guess in initial_guesses:
            # Dynamically select the optimization method based on progress
            method = 'L-BFGS-B' if self.evaluation_count < self.budget * 0.5 else 'Nelder-Mead'
            result = minimize(wrapped_func, guess, method=method, bounds=list(zip(func.bounds.lb, func.bounds.ub)))

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Iteratively refine bounds around the current best solution
            reduction_factor = (self.budget - self.evaluation_count) / self.budget
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - reduction_factor * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + reduction_factor * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))

            # If budget is exhausted, terminate
            if self.evaluation_count >= self.budget:
                break

            # Restart strategy to escape local optima
            if self.budget * 0.75 < self.evaluation_count <= self.budget * 0.9 and best_value < 0.01:
                # Generate new initial guesses using a scaled Sobol sequence
                new_sample_points = sampler.random_base2(m=2)  # Fewer points for a focused restart
                new_guesses = qmc.scale(new_sample_points, func.bounds.lb, func.bounds.ub).tolist()
                initial_guesses.extend(new_guesses)

        return best_solution