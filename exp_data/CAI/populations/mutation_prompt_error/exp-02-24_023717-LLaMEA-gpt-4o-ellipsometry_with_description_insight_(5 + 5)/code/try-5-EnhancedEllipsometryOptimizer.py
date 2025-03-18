import numpy as np
from scipy.optimize import minimize

class EnhancedEllipsometryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        # Initialize best solution and cost
        best_solution = None
        best_cost = float('inf')

        # Enhanced uniform initial sampling with more diverse samples
        num_initial_samples = max(10, self.budget // 3)
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        for initial_guess in initial_samples:
            cost = func(initial_guess)
            self.calls += 1
            if cost < best_cost:
                best_cost = cost
                best_solution = initial_guess
            if self.calls >= self.budget:
                return best_solution

        # Adaptive Nelder-Mead for local optimization
        remaining_budget = self.budget - self.calls
        if remaining_budget > 0:
            precision = 1e-6 if remaining_budget > 50 else 1e-4
            result = minimize(func, best_solution, method='Nelder-Mead',
                              bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                              options={'maxfev': remaining_budget, 'xatol': precision, 'fatol': precision})
            self.calls += result.nfev
            if result.fun < best_cost:
                best_cost = result.fun
                best_solution = result.x

        return best_solution