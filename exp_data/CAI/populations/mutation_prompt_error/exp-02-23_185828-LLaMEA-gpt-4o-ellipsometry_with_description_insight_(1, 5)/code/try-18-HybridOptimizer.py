import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(self.budget // 2, 20)
        
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        evaluations = 0

        # Evaluate initial samples and select the best half
        initial_values = []
        for sample in initial_samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            initial_values.append((value, sample))
            evaluations += 1

        initial_values.sort()  # Sort based on function values
        selected_samples = [x[1] for x in initial_values[:len(initial_values)//2]]  # Take the best half
        
        for sample in selected_samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            if value < best_value:
                best_value = value
                best_solution = sample
        
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            evaluations += 1
            return func(x)
        
        if evaluations < self.budget:
            perturbation_std = max(0.001, 0.01 * (1 - best_value / initial_samples.mean()))
            best_solution += np.random.normal(0, perturbation_std, self.dim)
            result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun

        return best_solution