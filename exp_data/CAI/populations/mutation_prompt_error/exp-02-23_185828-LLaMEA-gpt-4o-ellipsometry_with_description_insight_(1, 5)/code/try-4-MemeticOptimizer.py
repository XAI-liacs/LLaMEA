import numpy as np
from scipy.optimize import minimize

class MemeticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 3, 10)  # Sample initial points
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        evaluations = 0

        # Evaluate initial samples
        for sample in initial_samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            evaluations += 1
            return func(x)

        # Refine the best initial sample using Nelder-Mead
        if evaluations < self.budget:
            result = minimize(wrapped_func, best_solution, method='Nelder-Mead', bounds=bounds, options={'maxiter': self.budget - evaluations})
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun

        # Self-adaptive bounds adjustment
        iteration = 0
        while evaluations < self.budget:
            iteration += 1
            new_bounds = np.clip(best_solution + 0.1 * iteration, bounds[:, 0], bounds[:, 1]).T
            refined_solution = np.random.uniform(new_bounds[0], new_bounds[1], self.dim)
            value = wrapped_func(refined_solution)

            if value < best_value:
                best_value = value
                best_solution = refined_solution

        return best_solution