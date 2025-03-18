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
        initial_samples = min(self.budget // 8, 15)  # Increased to 15 samples
        for _ in range(initial_samples):
            initial_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            perturbed_guess = initial_guess + np.random.normal(0, 0.02 * np.std(initial_guess), self.dim)  # Adjusted perturbation scaling
            score = func(perturbed_guess)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = perturbed_guess

        # Step 2: Local optimization using BFGS
        while self.evaluations < self.budget:
            adaptive_options = {'maxiter': self.budget - self.evaluations, 'learning_rate': 0.03}  # Adjusted learning rate
            res = minimize(func, best_solution, bounds=bounds, method='L-BFGS-B', options=adaptive_options)
            self.evaluations += res.nfev
            if res.fun < best_score:
                best_score = res.fun
                best_solution = res.x
            else:  # Enhanced dynamic restart with gradient alignment
                restart_guess = best_solution + np.random.normal(0, 0.1, self.dim)
                res = minimize(func, restart_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget - self.evaluations})
                self.evaluations += res.nfev
                if res.fun < best_score:
                    best_score = res.fun
                    best_solution = res.x    
            if self.evaluations >= self.budget:
                break

        return best_solution