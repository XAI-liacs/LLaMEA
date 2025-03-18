import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

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

    def multi_start_strategy(self, func, bounds, num_starts=3):
        """Perform multiple local searches and keep the best result."""
        best_value = float('inf')
        best_solution = None
        for _ in range(num_starts):
            initial_guess = self.uniform_sampling(func.bounds)
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds)
            self.evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        return best_solution, best_value

    def __call__(self, func):
        # Initial sampling and multi-start
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        best_solution, best_value = self.multi_start_strategy(func, bounds)

        # Main optimization loop
        while self.evaluations < self.budget:
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds)
            self.evaluations += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Refine bounds around the best solution found
            bounds = self.refine_bounds(best_solution, func.bounds)

            # Dynamic learning rate adjustment and exploration
            learning_rate = 0.1 * (self.budget - self.evaluations) / self.budget
            best_solution = best_solution + learning_rate * np.random.randn(self.dim)
            if self.evaluations < self.budget:
                new_value = func(best_solution)
                self.evaluations += 1
                if new_value < best_value:
                    best_value = new_value
                    best_solution = best_solution

        return best_solution