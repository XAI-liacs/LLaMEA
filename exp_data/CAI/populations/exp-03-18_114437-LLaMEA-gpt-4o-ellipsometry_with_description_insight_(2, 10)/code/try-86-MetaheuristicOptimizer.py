import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Step 1: Enhanced initial sampling strategy with diversity consideration
        num_initial_samples = max(min(self.budget // 4, 15), 5)  # Adjusted line for initial sample count
        initial_samples = [
            np.random.uniform(func.bounds.lb, func.bounds.ub)
            for _ in range(num_initial_samples)
        ]
        initial_samples[np.argmin([func(sample) for sample in initial_samples])] = np.mean(
            initial_samples, axis=0)  # Diversity enhancement step
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

        # Step 3: Use BFGS for local optimization with a new adaptive bound strategy
        def objective(x):
            nonlocal evals_remaining
            if evals_remaining <= 0:
                return float('inf')
            evals_remaining -= 1
            return func(x)

        # Constrained optimization with adaptive bounds
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]

        result = minimize(
            objective, 
            x0=best_solution, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxfun': evals_remaining, 'ftol': 1e-10, 'maxls': 30}  # Changed termination criteria
        )

        # Step 4: Multistage refinement using combined Nelder-Mead and Powell method
        dynamic_threshold = np.std(scores) / 2.5
        if result.fun > best_score + dynamic_threshold and evals_remaining > 0:
            result = minimize(
                objective, 
                x0=result.x, 
                method='Nelder-Mead', 
                options={'maxfev': evals_remaining}
            )
            if result.fun > best_score and evals_remaining > 0:
                result = minimize(
                    objective, 
                    x0=result.x, 
                    method='Powell', 
                    bounds=bounds,
                    options={'maxfev': evals_remaining}
                )

        return result.x if result.success else best_solution