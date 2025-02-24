import numpy as np
from scipy.optimize import minimize

class AdaptiveSamplingDynamicConvergence:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        remaining_budget = self.budget
        best_solution = None
        best_value = float('inf')
        
        # Divide the budget for initial global exploration
        exploration_budget = int(self.budget * 0.45)  # Increased exploration budget
        convergence_budget = self.budget - exploration_budget

        # Step 1: Perform global exploration with enhanced adaptive uniform sampling
        adaptive_sample_size = max(8, min(20, exploration_budget // 3))  # Further increased sample size
        
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(adaptive_sample_size, self.dim))
        
        # Evaluate all initial samples
        sample_values = np.array([func(sample) for sample in samples])
        remaining_budget -= adaptive_sample_size

        # Find the best initial sample
        best_idx = np.argmin(sample_values)
        best_value = sample_values[best_idx]
        best_solution = samples[best_idx]

        # Step 2: Dynamic convergence using an enhanced local optimizer with iterative bound refinement
        current_bounds = bounds
        while remaining_budget > 0:
            result = minimize(func, best_solution, method='L-BFGS-B', bounds=current_bounds, options={'maxfun': min(convergence_budget, remaining_budget), 'disp': False})
            remaining_budget -= result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

                # Iteratively tighten bounds towards the current best solution using a more aggressive strategy
                tight_bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb))) for (x, (lb, ub)) in zip(best_solution, current_bounds)]
                current_bounds = tight_bounds

        return best_solution