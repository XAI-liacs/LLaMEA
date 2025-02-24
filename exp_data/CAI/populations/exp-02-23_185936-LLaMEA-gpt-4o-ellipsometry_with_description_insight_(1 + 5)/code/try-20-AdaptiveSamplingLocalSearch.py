import numpy as np
from scipy.optimize import minimize

class AdaptiveSamplingLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        initial_samples = min(self.budget // 10, 10)
        
        # Initial uniform sampling
        initial_guesses = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(initial_samples, self.dim))
        
        evaluations = 0
        best_solution = None
        best_value = float('inf')

        while evaluations < self.budget:
            for guess in initial_guesses:
                if evaluations >= self.budget:
                    break

                # Local optimization using BFGS
                result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evaluations})
                evaluations += result.nfev

                # Update best solution
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x

            # Adaptive sampling: generate new guesses based on best solutions found
            if evaluations < self.budget:
                perturb_radius = 0.1 * (np.array([b[1] for b in bounds]) - np.array([b[0] for b in bounds]))
                new_guesses = best_solution + np.random.uniform(-1, 1, (initial_samples, self.dim)) * perturb_radius
                new_guesses = np.clip(new_guesses, [b[0] for b in bounds], [b[1] for b in bounds])
                initial_guesses = new_guesses

        return best_solution