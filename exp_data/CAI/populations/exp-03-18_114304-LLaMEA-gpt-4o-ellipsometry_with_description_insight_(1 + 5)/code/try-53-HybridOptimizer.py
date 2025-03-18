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
        
        samples = np.random.uniform(lb, ub, (self.budget // 2, self.dim))
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return best_value
            value = func(x)
            evaluations += 1
            return value

        dynamic_bounds = [(max(lb[i], best_solution[i] - 0.1), min(ub[i], best_solution[i] + 0.1)) for i in range(self.dim)]

        # Adaptive BFGS with curvature information
        options = {'maxiter': self.budget - evaluations, 'disp': False, 'gtol': 1e-6}  # Changed tolerance for better convergence
        result = minimize(local_optimization, best_solution, method='BFGS', bounds=dynamic_bounds, options=options)

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        return best_solution, best_value