import numpy as np
from scipy.optimize import minimize

class EllipsometryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        # Initialize best solution and cost
        best_solution = None
        best_cost = float('inf')

        # Uniform initial sampling
        num_initial_samples = min(10, self.budget // 2)
        lb, ub = func.bounds.lb, func.bounds.ub
        for _ in range(num_initial_samples):
            initial_guess = np.random.uniform(lb, ub, self.dim)
            cost = func(initial_guess)
            self.calls += 1
            if cost < best_cost:
                best_cost = cost
                best_solution = initial_guess
            if self.calls >= self.budget:
                return best_solution

        # Use Nelder-Mead for local optimization
        remaining_budget = self.budget - self.calls
        if remaining_budget > 0:
            result = minimize(func, best_solution, method='Nelder-Mead',
                              bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                              options={'maxfev': remaining_budget})
            self.calls += result.nfev
            if result.fun < best_cost:
                best_cost = result.fun
                best_solution = result.x

        return best_solution