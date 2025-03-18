import numpy as np
from scipy.optimize import minimize

class StochasticSamplingLocalRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        best_solution = None
        best_value = float('inf')

        # Stochastic sampling for diverse initial guesses
        num_initial_guesses = min(10, self.budget // (2 * self.dim))
        initial_guesses = np.random.uniform(bounds[0], bounds[1], 
                                            (num_initial_guesses, self.dim))

        for guess in initial_guesses:
            # Perform a local optimization from the stochastic guess
            result = minimize(self.evaluate_func, guess, args=(func,),
                              method='L-BFGS-B', bounds=bounds.T,
                              options={'maxfun': (self.budget - self.evaluations) // num_initial_guesses})

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Break early if budget exceeded
            if self.evaluations >= self.budget:
                break

            # Dynamic local refinement: slightly perturb best solution found
            refinement_range = 0.05 * (bounds[1] - bounds[0])
            refinement_guess = best_solution + np.random.uniform(-refinement_range, refinement_range, self.dim)
            refinement_guess = np.clip(refinement_guess, bounds[0], bounds[1])

            # Perform a refined local search from the perturbed solution
            result = minimize(self.evaluate_func, refinement_guess, args=(func,),
                              method='L-BFGS-B', bounds=bounds.T,
                              options={'maxfun': (self.budget - self.evaluations) // num_initial_guesses})

            if result.success and result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
        return best_solution

    def evaluate_func(self, x, func):
        if self.evaluations < self.budget:
            value = func(x)
            self.evaluations += 1
            return value
        else:
            # Avoid further evaluations and terminate early
            return float('inf')