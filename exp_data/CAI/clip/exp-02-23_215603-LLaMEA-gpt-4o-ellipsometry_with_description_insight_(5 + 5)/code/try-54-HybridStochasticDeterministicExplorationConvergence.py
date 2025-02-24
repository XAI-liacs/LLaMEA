import numpy as np
from scipy.optimize import minimize

class HybridStochasticDeterministicExplorationConvergence:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        remaining_budget = self.budget
        best_solution = None
        best_value = float('inf')
        
        # Step 1: Stochastic exploration using Sobol sequence for evenly distributed samples
        exploration_budget = int(self.budget * 0.4)
        convergence_budget = self.budget - exploration_budget

        num_samples = min(16, exploration_budget // 2)  # Balanced number of initial samples
        samples = np.random.rand(num_samples, self.dim)
        samples = func.bounds.lb + samples * (func.bounds.ub - func.bounds.lb)

        # Evaluate all initial samples
        sample_values = np.array([func(sample) for sample in samples])
        remaining_budget -= num_samples

        # Find the best initial sample
        best_idx = np.argmin(sample_values)
        best_value = sample_values[best_idx]
        best_solution = samples[best_idx]

        # Step 2: Adaptive deterministic convergence using CMA-ES
        while remaining_budget > 0:
            result = minimize(func, best_solution, method='Trust-Constr', bounds=bounds, options={'maxiter': convergence_budget, 'disp': False})
            remaining_budget -= result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

                # Dynamically adjust bounds based on adaptive step-size approach
                step_size = (10**-result.nfev) * 0.1  # Adaptive step-size decreases with more evaluations
                tight_bounds = [(max(lb, x - step_size * (ub - lb)), min(ub, x + step_size * (ub - lb))) for (x, (lb, ub)) in zip(best_solution, bounds)]
                bounds = tight_bounds

        return best_solution

# Example usage:
# Assuming you have a function `func` with attributes `bounds.lb` and `bounds.ub`
# optimizer = HybridStochasticDeterministicExplorationConvergence(budget=100, dim=2)
# best_solution = optimizer(func)