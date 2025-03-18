import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 3, 50)  # Adjusted initial sampling size

        # Using Latin Hypercube Sampling for better initial exploration
        sampler = qmc.LatinHypercube(d=self.dim)
        initial_lhs_samples = sampler.random(n=num_initial_samples)
        initial_samples = qmc.scale(initial_lhs_samples, bounds[:, 0], bounds[:, 1])
        
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
        
        # Refine the best initial sample using L-BFGS-B
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            evaluations += 1
            return func(x)
        
        if evaluations < self.budget:
            perturbation_std = 0.005 + 0.02 * np.median(initial_samples, axis=0).mean()  # Enhanced perturbation precision
            best_solution += np.random.normal(0, perturbation_std, self.dim)
            # Enhanced convergence criteria
            result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-10})
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun

        return best_solution