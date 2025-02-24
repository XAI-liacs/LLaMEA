import numpy as np
from scipy.optimize import minimize

class MultiStartSequentialOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        local_bounds = [(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
        
        # Calculate number of random starts and iterations per local optimization
        num_starts = min(max(5, int(self.budget * 0.3)), self.budget // 4)
        local_iters = self.budget // num_starts

        best_solution = None
        best_value = float('inf')

        for _ in range(num_starts):
            # Randomly sample an initial guess within the bounds
            initial_guess = np.random.rand(self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            
            # Local optimization using L-BFGS-B
            result = minimize(
                func,
                initial_guess,
                method='L-BFGS-B',
                bounds=local_bounds,
                options={'maxfun': min(local_iters, self.budget - self.evals)}
            )
            
            # Update the number of evaluations
            self.evals += result.nfev
            
            # Check and update the best solution found so far
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Stop if the budget is exhausted
            if self.evals >= self.budget:
                break
        
        return best_solution