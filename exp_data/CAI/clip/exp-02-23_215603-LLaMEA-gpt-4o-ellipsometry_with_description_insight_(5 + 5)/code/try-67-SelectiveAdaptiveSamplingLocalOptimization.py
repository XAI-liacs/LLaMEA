import numpy as np
from scipy.optimize import minimize

class SelectiveAdaptiveSamplingLocalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        remaining_budget = self.budget
        best_solution = None
        best_value = float('inf')        

        # Step 1: Initial selective sampling
        num_samples = min(10, remaining_budget // 2)
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(num_samples, self.dim))
        sample_values = np.array([func(sample) for sample in samples])
        remaining_budget -= num_samples
        
        # Sort samples by function value
        sorted_indices = np.argsort(sample_values)
        samples = samples[sorted_indices]
        sample_values = sample_values[sorted_indices]
        
        # Select subset of samples using variance reduction
        selected_indices = sorted_indices[:max(1, num_samples // 3)]
        selected_samples = samples[selected_indices]
        selected_values = sample_values[selected_indices]

        # Step 2: Identify best sample for local optimization
        best_idx = np.argmin(selected_values)
        best_value = selected_values[best_idx]
        best_solution = selected_samples[best_idx]

        # Step 3: Local optimization with adaptive bounds
        current_bounds = bounds
        while remaining_budget > 0:
            # Perform local optimization
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=current_bounds, options={'maxfun': remaining_budget, 'disp': False})
            remaining_budget -= result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

                # Dynamically adjust bounds to hone in on best solution
                tight_bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb))) for (x, (lb, ub)) in zip(best_solution, current_bounds)]
                current_bounds = tight_bounds

        return best_solution

# Example usage:
# Assuming you have a function `func` with attributes `bounds.lb` and `bounds.ub`
# optimizer = SelectiveAdaptiveSamplingLocalOptimization(budget=100, dim=2)
# best_solution = optimizer(func)