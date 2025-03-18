import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Step 1: Uniformly sample initial points in the parameter space
        num_initial_samples = max(min(self.budget // 3, 20), 5)
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

        # Step 3: Use Multi-Start L-BFGS-B for local optimization
        def objective(x):
            nonlocal evals_remaining
            if evals_remaining <= 0:
                return float('inf')
            evals_remaining -= 1
            return func(x)

        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_restarts = min(3, evals_remaining // 10)  # New multi-start feature
        results = []

        for _ in range(num_restarts):
            result = minimize(
                objective, 
                x0=np.random.uniform(func.bounds.lb, func.bounds.ub), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxfun': evals_remaining, 'ftol': 1e-9, 'maxls': 15}  # Adjusted max line search steps
            )
            results.append(result)

        # Select the best result from the multiple L-BFGS-B runs
        best_result = min(results, key=lambda r: r.fun if r.success else float('inf'))

        # Refinement using Nelder-Mead based on dynamically adjusted L-BFGS-B result quality
        dynamic_threshold = np.std(scores) / 2  # Adjusted threshold calculation
        if best_result.fun > best_score + dynamic_threshold and evals_remaining > 0:
            best_result = minimize(
                objective, 
                x0=best_result.x, 
                method='Nelder-Mead', 
                options={'maxfev': evals_remaining}
            )

        return best_result.x if best_result.success else best_solution