import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        # Adaptive initial exploration with uniform random sampling
        initial_sample_size = min(10, self.budget // 3)  # Adjusted sample size
        samples = np.random.uniform(lb, ub, (initial_sample_size, self.dim))
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Local exploitation using BFGS starting from the best sample found
        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return best_value  # Return the best value found if budget is exhausted
            value = func(x)
            evaluations += 1
            return value

        # Run BFGS from the best initial sample
        result = minimize(local_optimization, best_solution, method='BFGS',
                          bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                          options={'maxiter': self.budget - evaluations, 'disp': False})

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        return best_solution, best_value