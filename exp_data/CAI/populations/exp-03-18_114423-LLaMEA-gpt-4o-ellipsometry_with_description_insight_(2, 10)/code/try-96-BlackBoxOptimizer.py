import numpy as np
from scipy.optimize import minimize

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

        while remaining_budget > 0:
            num_samples = max(3, int(remaining_budget / self.budget * 20))
            initial_guesses = [np.array([np.random.uniform(low, high) for low, high in bounds]) for _ in range(num_samples)]
            weights = np.linspace(1, num_samples, num_samples)
            initial_guess = np.average(initial_guesses, axis=0, weights=weights)
            
            local_budget = max(5, remaining_budget // 3 + int(previous_improvement < 0.05))
            # Change: Adjusted calculation of dynamic_ftol
            dynamic_ftol = 1e-6 * (1 + ((self.budget - remaining_budget) / self.budget) ** 0.5)
            result = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'ftol': dynamic_ftol, 'maxfun': local_budget})

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                previous_improvement = best_value
            elif remaining_budget < self.budget // 4:
                remaining_budget = self.budget // 2

            remaining_budget -= result.nfev

        return best_solution, best_value