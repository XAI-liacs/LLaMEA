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

        # Define a wrapping function to count evaluations
        def wrapped_func(x):
            if self.evaluation_count >= self.budget:
                return float('inf')
            self.evaluation_count += 1
            return func(x)

        # Adaptive bounds tightening factor
        tightening_factor = 0.05

        # Begin with a local optimizer
        for guess in initial_guesses:
            # Dynamically select the optimization method
            if self.evaluation_count < self.budget / 3:
                method = 'L-BFGS-B'
            elif self.evaluation_count < 2 * self.budget / 3:
                method = 'Nelder-Mead'
            else:
                method = 'L-BFGS-B'

            result = minimize(wrapped_func, guess, method=method, bounds=list(zip(func.bounds.lb, func.bounds.ub)))

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Update bounds based on the current best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - tightening_factor * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + tightening_factor * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))

            # Gradually increase the tightening factor
            tightening_factor *= 1.1

            # If budget is exhausted, terminate
            if self.evaluation_count >= self.budget:
                break

        return best_solution