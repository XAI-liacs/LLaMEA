import numpy as np
from scipy.optimize import minimize

class EnhancedAdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        initial_guess = self.dynamic_sampling(bounds, self.budget)
        remaining_budget = self.budget - len(initial_guess)

        best_solution = None
        best_cost = float('inf')

        for guess in initial_guess:
            result = self.local_optimization(func, guess, bounds, remaining_budget)
            if result.fun < best_cost:
                best_solution = result.x
                best_cost = result.fun
                remaining_budget -= result.nfev

            if remaining_budget <= 0:
                break

        return best_solution

    def dynamic_sampling(self, bounds, budget):
        num_samples = max(4, budget // 10)  # Adjust sample size based on budget
        samples = []
        for _ in range(num_samples):
            sample = [np.random.uniform(low, high) for low, high in bounds]
            samples.append(sample)
        return samples

    def local_optimization(self, func, initial_guess, bounds, remaining_budget):
        # Refine the bounds based on initial_guess for tighter search
        refined_bounds = [(max(low, guess - (high - low) * 0.1), min(high, guess + (high - low) * 0.1))
                          for (low, high), guess in zip(bounds, initial_guess)]
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(func, initial_guess, method='L-BFGS-B', bounds=refined_bounds, options=options)
        return result