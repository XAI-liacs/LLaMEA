import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        # Initial exploration with uniform random sampling
        num_initial_samples = self.budget // 3
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Local exploitation using BFGS
        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return best_value
            value = func(x)
            evaluations += 1
            return value

        dynamic_bounds = [(max(lb[i], best_solution[i] - 0.05), min(ub[i], best_solution[i] + 0.05)) for i in range(self.dim)]
        
        # Multi-start BFGS with adjusted dynamic bounds
        for _ in range(2):  # Attempt BFGS from two different starting points
            result = minimize(local_optimization, best_solution, method='BFGS',
                              bounds=dynamic_bounds,
                              options={'maxiter': self.budget - evaluations, 'disp': False})
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            dynamic_bounds = [(max(lb[i], result.x[i] - 0.05), min(ub[i], result.x[i] + 0.05)) for i in range(self.dim)]

        return best_solution, best_value