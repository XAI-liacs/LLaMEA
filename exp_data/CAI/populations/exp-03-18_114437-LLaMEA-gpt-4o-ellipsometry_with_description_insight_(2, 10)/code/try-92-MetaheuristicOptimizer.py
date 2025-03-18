import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        num_initial_samples = max(min(self.budget // (2 + self.dim // 5), 25), 5)
        initial_samples = [
            np.random.uniform(func.bounds.lb, func.bounds.ub)
            for _ in range(num_initial_samples)
        ]
        evals_remaining = self.budget - num_initial_samples
        best_solution = None
        best_score = float('inf')

        scores = []
        for sample in initial_samples:
            score = func(sample)
            self.evaluations += 1
            scores.append(score)
            if score < best_score:
                best_score = score
                best_solution = sample

        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]

        random_exploration_factor = 0.1  # Added stochastic exploration phase
        def objective(x):
            nonlocal evals_remaining
            penalty = 0.05 * np.sum(np.log1p(np.abs(x - best_solution)))
            if evals_remaining <= 0:
                return float('inf')
            evals_remaining -= 1
            random_exploration = np.random.uniform(-random_exploration_factor, random_exploration_factor, len(x))
            return func(x + random_exploration) + penalty

        result = minimize(
            objective, 
            x0=best_solution, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxfun': evals_remaining, 'ftol': 1e-8, 'maxls': 15}  # Adjusted line search
        )

        dynamic_threshold = np.std(scores) / 3.5
        if result.fun > best_score + dynamic_threshold and evals_remaining > 0:
            adaptive_step = np.std(result.x) * 0.2  # Adaptive gradient scaling
            result = minimize(
                objective, 
                x0=result.x * (1 + adaptive_step),
                method='Nelder-Mead', 
                options={'maxfev': evals_remaining}
            )

        return result.x if result.success else best_solution