import numpy as np
from scipy.optimize import minimize

class HybridLocalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        best_solution = None
        best_score = float('inf')
        
        # Step 1: Uniform random sampling for initial guesses
        initial_samples = max(2, self.budget // 15)  # Adjusted sampling rate for better exploration
        for _ in range(initial_samples):
            initial_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            score = func(initial_guess)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = initial_guess

        # Step 2: Local optimization using BFGS
        while self.evaluations < self.budget:
            res = minimize(func, best_solution, bounds=bounds, method='L-BFGS-B')
            self.evaluations += res.nfev
            if res.fun < best_score:
                best_score = res.fun
                best_solution = res.x
            if self.evaluations >= self.budget:
                break

        return best_solution