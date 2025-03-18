import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalSearchOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        # Start with a small number of initial samples
        num_initial_samples = max(self.budget // 5, 5)

        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
        
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
        
        # Dynamically adjust the sampling density based on remaining budget
        while evaluations < self.budget:
            if evaluations >= self.budget:
                break
            
            def wrapped_func(x):
                nonlocal evaluations
                if evaluations >= self.budget:
                    return float('inf')
                evaluations += 1
                return func(x)
            
            # Use a refined local optimizer for the best solution found
            result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun
            
            # If good convergence is observed, reduce the number of new samples
            if result.success and result.nfev > (self.budget // 10):
                break
            else:
                num_new_samples = (self.budget - evaluations) // 10
                new_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_new_samples, self.dim))
                for sample in new_samples:
                    if evaluations >= self.budget:
                        break
                    value = func(sample)
                    evaluations += 1
                    if value < best_value:
                        best_value = value
                        best_solution = sample

        return best_solution