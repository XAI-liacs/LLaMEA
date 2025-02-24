import numpy as np
from scipy.optimize import minimize

class RefinedAdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        initial_guess = self.dynamic_sampling(bounds)
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

    def dynamic_sampling(self, bounds):
        num_samples = max(self.budget // 10, 5)  # Dynamic sampling: 10% of budget or at least 5 samples.
        samples = []
        for _ in range(num_samples):
            sample = [np.random.uniform(low, high) for low, high in bounds]  # Uniform distribution for broad coverage.
            samples.append(sample)
        return samples

    def local_optimization(self, func, initial_guess, bounds, remaining_budget):
        options = {'maxiter': min(remaining_budget, 100), 'disp': False}  # Limit iterations to 100 for more restarts.
        result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options=options)
        return result