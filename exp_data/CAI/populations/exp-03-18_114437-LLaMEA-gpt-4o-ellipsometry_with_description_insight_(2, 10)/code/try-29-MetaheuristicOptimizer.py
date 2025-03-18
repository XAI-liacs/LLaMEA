import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Step 1: Use Sobol sequences for better initial coverage
        num_initial_samples = min(self.budget // 2, 10)
        sobol_sampler = Sobol(d=self.dim)
        initial_samples = sobol_sampler.random_base2(m=int(np.log2(num_initial_samples))) * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
        evals_remaining = self.budget - num_initial_samples
        best_solution = None
        best_score = float('inf')

        # Step 2: Evaluate initial samples and select the best one
        scores = []
        for sample in initial_samples:
            result = minimize(func, sample, method='L-BFGS-B', bounds=[(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
            score = result.fun
            scores.append(score)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = result.x

        # Calculate adaptive ftol based on score variability
        adaptive_ftol = max(1e-9, np.std(scores) / 100)

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
            options={'maxfun': evals_remaining, 'ftol': adaptive_ftol}  # Adjusted precision
        )

        return result.x if result.success else best_solution