import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class AdaptiveConvergentSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluation_count = 0

    def __call__(self, func):
        # Hybrid initial sampling for better coverage: Sobol and Latin Hypercube
        sampler_sobol = qmc.Sobol(d=self.dim, scramble=True)
        sampler_lhs = qmc.LatinHypercube(d=self.dim)
        sample_points_sobol = sampler_sobol.random_base2(m=3)  # 8 points Sobol
        sample_points_lhs = sampler_lhs.random(n=8)  # 8 points LHS
        sample_points = np.vstack((sample_points_sobol, sample_points_lhs))
        initial_guesses = qmc.scale(sample_points, func.bounds.lb, func.bounds.ub).tolist()
        
        best_solution = None
        best_value = float('inf')

        # Define a wrapping function to count evaluations
        def wrapped_func(x):
            if self.evaluation_count >= self.budget:
                return float('inf')
            self.evaluation_count += 1
            return func(x)

        # Begin with a local optimizer
        for guess in initial_guesses:
            # Probabilistic selection of optimization method
            method = 'L-BFGS-B' if np.random.rand() > 0.3 else 'Nelder-Mead'
            result = minimize(wrapped_func, guess, method=method, bounds=list(zip(func.bounds.lb, func.bounds.ub)))

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Update bounds based on the current best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (np.array(func.bounds.ub) - np.array(func.bounds.lb)))

            # Introduce Gaussian perturbation for restart strategy
            if self.evaluation_count < self.budget * 0.75:
                guess = best_solution + np.random.normal(0, 0.01, self.dim)

            # If budget is exhausted, terminate
            if self.evaluation_count >= self.budget:
                break

        return best_solution