import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        best_solution = None
        best_value = np.inf
        remaining_budget = self.budget

        while remaining_budget > 0:
            # Enhanced initial guesses using Sobol sequence
            sobol = qmc.Sobol(d=self.dim, scramble=True)
            initial_guesses = sobol.random_base2(m=2)  # Generates 4 samples
            initial_guesses = [
                func.bounds.lb + (func.bounds.ub - func.bounds.lb) * g
                for g in initial_guesses
            ]
            initial_guess = min(initial_guesses, key=lambda g: func(g))
            
            local_budget = max(5, remaining_budget // 2)
            
            # Hybrid strategy: perturbation and constrained local search
            perturbed_guess = initial_guess + np.random.normal(0, 0.1, self.dim)
            perturbed_guess = np.clip(perturbed_guess, func.bounds.lb, func.bounds.ub)

            result = minimize(func, perturbed_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': local_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            remaining_budget -= result.nfev

        return best_solution, best_value