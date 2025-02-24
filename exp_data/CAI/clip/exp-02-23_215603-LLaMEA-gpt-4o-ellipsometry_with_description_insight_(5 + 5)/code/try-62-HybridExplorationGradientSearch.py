import numpy as np
from scipy.optimize import minimize

class HybridExplorationGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        remaining_budget = self.budget
        best_solution = None
        best_value = float('inf')
        
        # Divide the budget for initial exploration and dynamic phases
        exploration_budget = max(5, int(self.budget * 0.3))
        dynamic_budget = self.budget - exploration_budget

        # Step 1: Initial stochastic exploration with uniform sampling
        samples = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(exploration_budget, self.dim))
        
        sample_values = np.array([func(sample) for sample in samples])
        remaining_budget -= exploration_budget

        # Find the best initial sample
        best_idx = np.argmin(sample_values)
        best_value = sample_values[best_idx]
        best_solution = samples[best_idx]

        # Step 2: Hybrid exploration-exploitation phase
        while remaining_budget > 0:
            if remaining_budget > dynamic_budget * 0.5:
                # Exploration phase: Use random sampling to potentially escape local optima
                new_sample = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=self.dim)
                new_value = func(new_sample)
                remaining_budget -= 1
            else:
                # Exploitation phase: Use gradient-based optimizer
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': min(dynamic_budget, remaining_budget), 'disp': False})
                new_sample = result.x
                new_value = result.fun
                remaining_budget -= result.nfev

            # Update the best solution found
            if new_value < best_value:
                best_value = new_value
                best_solution = new_sample

        return best_solution

# Example usage:
# Assuming you have a function `func` with attributes `bounds.lb` and `bounds.ub`
# optimizer = HybridExplorationGradientSearch(budget=100, dim=2)
# best_solution = optimizer(func)