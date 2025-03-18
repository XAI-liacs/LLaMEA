import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        best_solution = None
        best_value = float('inf')

        # Uniformly sample a fraction of the budget for initial guesses
        num_initial_samples = max(1, self.budget // 10)
        initial_samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_initial_samples, self.dim))

        for sample in initial_samples:
            if self.evaluations >= self.budget:
                break
            mutated_sample = self.mutate(sample, bounds)
            solution, value = self.local_search(func, mutated_sample, bounds)
            if value < best_value:
                best_solution, best_value = solution, value

        return best_solution

    def local_search(self, func, initial_point, bounds):
        if self.evaluations >= self.budget:
            return initial_point, func(initial_point)

        # Use a local optimizer (BFGS) for fast convergence with adaptive learning rate
        result = minimize(func, initial_point, method='L-BFGS-B', bounds=bounds, options={'learning_rate': 'adaptive', 'maxfun': self.budget - self.evaluations})
        self.evaluations += result.nfev

        return result.x, result.fun
    
    def mutate(self, sample, bounds):
        # Differential Evolution's mutation strategy
        F = 0.5 if np.random.rand() < 0.5 else 0.8
        noise = F * (np.random.uniform(-1, 1, self.dim))
        return np.clip(sample + noise, bounds[:, 0], bounds[:, 1])