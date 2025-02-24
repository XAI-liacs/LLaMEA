import numpy as np
from scipy.optimize import minimize

class MultiStartAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(5, self.budget // 4)
        
        # Step 1: Multi-Start Initial Sampling
        def multi_start_sampling():
            population = np.random.uniform(
                low=func.bounds.lb, 
                high=func.bounds.ub, 
                size=(num_initial_samples, self.dim)
            )
            f_values = np.array([func(ind) for ind in population])
            best_idx = np.argmin(f_values)
            return population[best_idx], f_values[best_idx]
        
        best_initial_sample, best_initial_value = multi_start_sampling()
        remaining_budget = self.budget - num_initial_samples
        
        # Step 2: Adaptive Local Optimization using BFGS
        if remaining_budget > 0:
            def local_objective(x):
                return func(x)
            
            starting_points = [best_initial_sample] + [
                np.random.uniform(low=func.bounds.lb, high=func.bounds.ub) for _ in range(3)
            ]
            
            best_result = {'fun': float('inf')}
            for start in starting_points:
                result = minimize(
                    local_objective, 
                    start, 
                    method='BFGS',
                    bounds=bounds if self.dim > 1 else None,
                    options={'maxiter': remaining_budget // len(starting_points), 'gtol': 1e-8}
                )
                if result.fun < best_result['fun']:
                    best_result = result

            if best_result['fun'] < best_initial_value:
                return best_result.x
        
        return best_initial_sample

# Example usage:
# Assume func is a black-box function with attributes bounds.lb and bounds.ub
# optimizer = MultiStartAdaptiveOptimizer(budget=100, dim=2)
# best_parameters = optimizer(func)