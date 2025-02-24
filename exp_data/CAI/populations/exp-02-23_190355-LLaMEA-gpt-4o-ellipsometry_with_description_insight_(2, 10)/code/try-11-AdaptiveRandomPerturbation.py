import numpy as np
from scipy.optimize import minimize

class AdaptiveRandomPerturbation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize bounds
        bounds = np.array([func.bounds.lb, func.bounds.ub])

        # Initial random guess within bounds
        initial_guess = np.random.uniform(bounds[0], bounds[1], self.dim)

        # Track function evaluations
        self.func_evaluations = 0
        def callback(x):
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                raise StopIteration("Budget exhausted")

        # Define a random perturbation function
        def random_perturbation(x, scale=0.05):
            return x + np.random.uniform(-scale, scale, size=x.shape)

        # Perform initial optimization using Nelder-Mead
        try:
            result = minimize(func, initial_guess, method='Nelder-Mead',
                              bounds=bounds.T, callback=callback,
                              options={'maxiter': self.budget, 'disp': False})
        except StopIteration:
            pass

        # Iteratively refine the solution with random perturbations
        current_solution = result.x
        while self.func_evaluations < self.budget:
            perturbed_solution = random_perturbation(current_solution)
            try:
                result = minimize(func, perturbed_solution, method='Nelder-Mead',
                                  bounds=bounds.T, callback=callback,
                                  options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
            except StopIteration:
                break
            current_solution = result.x

            # Update bounds to focus on the neighborhood of the current solution
            new_bounds = np.array([np.maximum(current_solution - 0.1 * (bounds[1] - bounds[0]), bounds[0]),
                                   np.minimum(current_solution + 0.1 * (bounds[1] - bounds[0]), bounds[1])])
            bounds = new_bounds

        return current_solution