import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 8, 12)  # Adjusted number of initial guesses
        
        # Adaptive sampling for initial points
        initial_guesses = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0
        epsilon = 1e-6  # Convergence threshold

        for guess in initial_guesses:
            if evaluations >= self.budget:
                break

            # Use the BFGS algorithm for local optimization
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evaluations, 'ftol': epsilon})
            evaluations += result.nfev

            # Update the best solution found so far
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                if best_value < epsilon:  # Early stopping condition
                    break
        
        # Dynamic tuning step
        if evaluations < self.budget:
            refined_result = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.budget - evaluations, 'xtol': epsilon})
            if refined_result.fun < best_value:
                best_solution = refined_result.x

        return best_solution