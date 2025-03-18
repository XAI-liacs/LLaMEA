import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Step 1: Uniformly sample initial points in the parameter space
        num_initial_samples = max(5, self.budget // 4)  # Dynamic number of samples
        initial_samples = [
            np.random.uniform(func.bounds.lb, func.bounds.ub)
            for _ in range(num_initial_samples)
        ]
        evals_remaining = self.budget - num_initial_samples
        best_solution = None
        best_score = float('inf')

        # Step 2: Evaluate initial samples and select the best one
        for sample in initial_samples:
            # Gradient-based initialization using L-BFGS-B
            result = minimize(func, sample, method='L-BFGS-B', bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
            score = result.fun
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = result.x

        # Step 3: Use BFGS for local optimization starting from the best initial sample
        def objective(x):
            nonlocal evals_remaining
            if evals_remaining <= 0:
                return float('inf')
            evals_remaining -= 1
            return func(x)

        # Constrained optimization to respect bounds
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]

        result = minimize(
            objective, 
            x0=best_solution, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxfun': evals_remaining}
        )

        return result.x if result.success else best_solution