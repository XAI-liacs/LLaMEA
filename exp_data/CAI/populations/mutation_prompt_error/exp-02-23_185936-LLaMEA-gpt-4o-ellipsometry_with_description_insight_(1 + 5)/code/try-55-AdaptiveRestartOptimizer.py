import numpy as np
from scipy.optimize import minimize

class AdaptiveRestartOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 15, 5)  # Fewer initial guesses to allow for restarts
        
        # Uniformly sample initial points within the defined bounds
        initial_guesses = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0
        epsilon = 1e-6  # Convergence threshold

        while evaluations < self.budget:
            for guess in initial_guesses:
                if evaluations >= self.budget:
                    break

                # Use the Nelder-Mead algorithm for local optimization
                result = minimize(func, guess, method='Nelder-Mead', bounds=bounds, options={'maxfev': self.budget - evaluations, 'fatol': epsilon})
                evaluations += result.nfev

                # Update the best solution found so far
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
                    if best_value < epsilon:  # Early stopping condition
                        return best_solution

            # Adaptive restarts: generate new initial guesses based on current best solution
            initial_guesses = np.array([best_solution + np.random.uniform(-0.1, 0.1, size=self.dim) for _ in range(num_initial_guesses)])
            initial_guesses = np.clip(initial_guesses, [b[0] for b in bounds], [b[1] for b in bounds])

        return best_solution