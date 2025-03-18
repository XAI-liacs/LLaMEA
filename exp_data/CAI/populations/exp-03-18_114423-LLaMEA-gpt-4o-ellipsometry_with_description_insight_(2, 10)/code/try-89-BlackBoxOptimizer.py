import numpy as np
from scipy.optimize import minimize, differential_evolution

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
        previous_improvement = np.inf
        stagnation_threshold = 0.01  # Stagnation threshold for triggering resampling

        while remaining_budget > 0:
            num_samples = max(3, remaining_budget // self.budget * 10)
            initial_guesses = [np.array([np.random.uniform(low, high) for low, high in bounds]) for _ in range(num_samples)]
            initial_guess = min(initial_guesses, key=lambda g: func(g))

            if previous_improvement < stagnation_threshold:
                num_samples *= 3  # Triple samples on stagnation

            local_budget = max(5, remaining_budget // 2 + int(previous_improvement < 0.05))

            # Use differential evolution for larger initial sample diversity
            if func(initial_guess) > best_value:
                result = differential_evolution(func, bounds, maxiter=local_budget // 5)
            else:
                result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-6 * (remaining_budget / self.budget), 'maxfun': local_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                previous_improvement = best_value
            elif remaining_budget < self.budget // 4:
                remaining_budget = self.budget // 2

            remaining_budget -= result.nfev
            stagnation_threshold *= 0.95  # Adjust stagnation threshold dynamically

        return best_solution, best_value