import numpy as np
from scipy.optimize import minimize

class EnhancedExplorationConvergenceRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        remaining_budget = self.budget
        best_solution = None
        best_value = float('inf')
        
        # Allocate budget for initial exploration
        exploration_budget = int(self.budget * 0.5)
        convergence_budget = self.budget - exploration_budget

        # Step 1: Perform global exploration with strategic sampling
        adaptive_sample_size = max(6, min(18, exploration_budget // 2))
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(adaptive_sample_size, self.dim))
        
        # Evaluate initial samples
        sample_values = np.array([func(sample) for sample in samples])
        remaining_budget -= adaptive_sample_size

        # Find the best sample
        best_idx = np.argmin(sample_values)
        best_value = sample_values[best_idx]
        best_solution = samples[best_idx]

        # Step 2: Gradient-based convergence with dynamic bounds
        current_bounds = bounds
        while remaining_budget > 0:
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=current_bounds, options={'maxfun': min(convergence_budget, remaining_budget), 'disp': False})
            remaining_budget -= result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

                # Tighten bounds dynamically towards the current best solution
                tight_bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb))) for (x, (lb, ub)) in zip(best_solution, current_bounds)]
                current_bounds = tight_bounds

        return best_solution

# Example usage:
# Assuming you have a function `func` with attributes `bounds.lb` and `bounds.ub`
# optimizer = EnhancedExplorationConvergenceRefinement(budget=100, dim=2)
# best_solution = optimizer(func)