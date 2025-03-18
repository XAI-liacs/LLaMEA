import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.momentum = np.zeros(dim)  # Initialize momentum

    def uniform_sampling(self, bounds):
        """Generate an initial guess by uniformly sampling the search space."""
        return [np.random.uniform(low, high) for low, high in zip(bounds.lb, bounds.ub)]

    def refine_bounds(self, current_best, bounds, factor=0.5):
        """Shrink the bounds around the current best solution."""
        dynamic_factor = factor * (1 - self.evaluations / self.budget)  # Dynamically adjust factor
        new_bounds = []
        for i in range(self.dim):
            center = current_best[i]
            half_range = (bounds.ub[i] - bounds.lb[i]) * dynamic_factor / 2
            new_low = max(bounds.lb[i], center - half_range)
            new_high = min(bounds.ub[i], center + half_range)
            new_bounds.append((new_low, new_high))
        return new_bounds

    def __call__(self, func):
        # Initial sampling
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        best_solution = self.uniform_sampling(func.bounds)
        best_value = func(best_solution)
        self.evaluations += 1

        # Main optimization loop
        while self.evaluations < self.budget:
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds)
            self.evaluations += result.nfev

            # Apply momentum to the solution update
            self.momentum = 0.9 * self.momentum + 0.1 * (result.x - best_solution)
            best_solution += self.momentum  # Update solution with momentum

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Refine bounds around the best solution found
            bounds = self.refine_bounds(best_solution, func.bounds)

            # If budget allows, perform another optimization with new bounds
            if self.evaluations < self.budget:
                new_guess = self.uniform_sampling(func.bounds)
                new_value = func(new_guess)
                self.evaluations += 1
                if new_value < best_value:
                    best_value = new_value
                    best_solution = new_guess
                    bounds = self.refine_bounds(best_solution, func.bounds)

        return best_solution