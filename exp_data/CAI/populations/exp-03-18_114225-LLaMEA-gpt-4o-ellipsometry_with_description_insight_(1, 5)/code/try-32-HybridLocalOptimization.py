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
        initial_samples = max(self.budget // 10, 5)  # Adjusted to 10% of budget or at least 5 samples
        for _ in range(initial_samples):
            initial_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            perturbed_guess = initial_guess + np.random.normal(0, 0.01 + (self.budget - self.evaluations) / (30 * self.budget), self.dim)  # Adjusted perturbation
            score = func(perturbed_guess)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = perturbed_guess

        # Step 2: Local optimization using BFGS
        adaptive_factor = 0.3  # Added adaptive factor for dynamic restart
        while self.evaluations < self.budget:
            res = minimize(func, best_solution, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget - self.evaluations})
            self.evaluations += res.nfev
            if res.fun < best_score:
                best_score = res.fun
                best_solution = res.x
            elif np.random.rand() < adaptive_factor:  # Adaptive restart frequency
                restart_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
                res = minimize(func, restart_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget - self.evaluations})
                self.evaluations += res.nfev
                if res.fun < best_score:
                    best_score = res.fun
                    best_solution = res.x    
            
            if self.evaluations >= self.budget:
                break

        return best_solution