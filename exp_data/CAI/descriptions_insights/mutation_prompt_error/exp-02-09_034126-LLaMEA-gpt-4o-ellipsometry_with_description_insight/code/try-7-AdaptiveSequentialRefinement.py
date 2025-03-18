import numpy as np
from scipy.optimize import minimize

class AdaptiveSequentialRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        initial_guess = self.uniform_sampling(bounds)
        remaining_budget = self.budget - len(initial_guess)
        
        best_solution = None
        best_cost = float('inf')

        for guess in initial_guess:
            result, new_bounds = self.local_optimization(func, guess, bounds, remaining_budget)
            if result.fun < best_cost:
                best_solution = result.x
                best_cost = result.fun
                remaining_budget -= result.nfev
                bounds = new_bounds  # Tighten bounds

            if remaining_budget <= 0:
                break

        return best_solution

    def uniform_sampling(self, bounds):
        num_samples = min(self.budget // 8, 10)
        samples = []
        for _ in range(num_samples):
            sample = [np.random.uniform(low, high) for low, high in bounds]
            samples.append(sample)
        return samples

    def local_optimization(self, func, initial_guess, bounds, remaining_budget):
        options = {'maxiter': remaining_budget, 'disp': False}
        result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options=options)

        # Tighten bounds based on the solution found
        new_bounds = []
        for i, (low, high) in enumerate(bounds):
            center = result.x[i]
            radius = (high - low) / 4  # Reduce the search space
            new_low = max(low, center - radius)
            new_high = min(high, center + radius)
            new_bounds.append((new_low, new_high))

        return result, new_bounds