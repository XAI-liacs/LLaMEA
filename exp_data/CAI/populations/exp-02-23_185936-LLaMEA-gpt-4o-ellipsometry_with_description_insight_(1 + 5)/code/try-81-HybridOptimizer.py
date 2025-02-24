import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 10, 10)
        
        # Use Sobol sequence for better initial point coverage
        sobol_engine = Sobol(d=self.dim, scramble=False)
        initial_guesses = sobol_engine.random_base2(m=int(np.log2(num_initial_guesses)))
        initial_guesses = initial_guesses * (np.array([b[1] for b in bounds]) - np.array([b[0] for b in bounds])) + np.array([b[0] for b in bounds])

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

        return best_solution