import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Step 1: Adaptive sampling rate based on dimensionality
        num_initial_samples = max(min(self.budget // (2 + self.dim // 4), 25), 5)  # Slightly adjusted adaptive sampling rate
        initial_samples = [
            np.random.uniform(func.bounds.lb, func.bounds.ub)
            for _ in range(num_initial_samples)
        ]
        evals_remaining = self.budget - num_initial_samples
        best_solution = None
        best_score = float('inf')

        # Step 2: Evaluate initial samples and select the best one
        scores = []
        for sample in initial_samples:
            score = func(sample)
            self.evaluations += 1
            scores.append(score)
            if score < best_score:
                best_score = score
                best_solution = sample

        # Step 3: Use BFGS for local optimization starting from the best initial sample
        def objective(x):
            nonlocal evals_remaining
            penalty = 0.04 * np.sum(np.log1p(np.abs(x - best_solution)))  # Fine-tuned dynamic penalty adjustment
            if evals_remaining <= 0:
                return float('inf')
            evals_remaining -= 1
            return func(x) + penalty  # Adjusted objective

        # Constrained optimization to respect bounds
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]

        result = minimize(
            objective, 
            x0=best_solution, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxfun': evals_remaining, 'ftol': 1e-8, 'maxls': 20}
        )

        # Refinement using Nelder-Mead based on dynamically adjusted L-BFGS-B result quality
        dynamic_threshold = np.std(scores) / 3.5
        if result.fun > best_score + dynamic_threshold and evals_remaining > 0:
            result = minimize(
                objective, 
                x0=result.x, 
                method='Nelder-Mead', 
                options={'maxfev': evals_remaining}
            )

        return result.x if result.success else best_solution