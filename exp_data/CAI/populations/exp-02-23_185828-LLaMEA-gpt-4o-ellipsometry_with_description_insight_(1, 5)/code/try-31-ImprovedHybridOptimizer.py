import numpy as np
from scipy.optimize import minimize
from pyDOE import lhs

class ImprovedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 2, 50)

        # Use Latin Hypercube Sampling for better initial coverage
        initial_samples = lhs(self.dim, samples=num_initial_samples)
        scaled_samples = initial_samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        
        best_solution = None
        best_value = float('inf')
        
        evaluations = 0

        # Evaluate initial samples
        for sample in scaled_samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample
        
        # Refine the best initial sample using Trust-Region method
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            evaluations += 1
            return func(x)
        
        if evaluations < self.budget:
            perturbation_std = 0.01 + 0.02 * (1 - best_value / scaled_samples.mean())
            best_solution += np.random.normal(0, perturbation_std, self.dim)
            result = minimize(wrapped_func, best_solution, method='trust-constr', bounds=bounds, options={'gtol': 1e-6})
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun

        return best_solution