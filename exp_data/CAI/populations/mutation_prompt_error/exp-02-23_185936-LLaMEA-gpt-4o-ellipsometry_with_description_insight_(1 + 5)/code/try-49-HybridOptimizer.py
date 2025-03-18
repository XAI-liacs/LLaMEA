import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 10, 10)
        
        # Uniformly sample initial points within the defined bounds
        initial_guesses = np.array([np.random.uniform(low=b[0], high=b[1], size=self.dim) for b in [bounds] * num_initial_guesses])

        best_solution = None
        best_value = float('inf')
        evaluations = 0
        epsilon = 1e-6  # Convergence threshold
        adapt_epsilon = epsilon * 0.1  # Adaptive adjustment

        def optimize_local(guess):
            nonlocal evaluations, best_solution, best_value
            result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - evaluations, 'ftol': adapt_epsilon})
            evaluations += result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(optimize_local, guess) for guess in initial_guesses]
            for future in futures:
                if evaluations >= self.budget or best_value < epsilon:
                    break

        return best_solution