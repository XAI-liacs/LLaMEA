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
        initial_samples = min(self.budget // 8, 10)  # Limited to 12.5% of budget or 10 samples
        for _ in range(initial_samples):
            initial_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            # Slight adjustment to perturbation scaling to enhance exploration
            scale_factor = 0.02 * np.std(initial_guess) * (1 - self.evaluations / self.budget) * np.log1p(self.budget)  # Modified scale factor
            perturbed_guess = initial_guess + np.random.normal(0, scale_factor, self.dim)
            score = func(perturbed_guess)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = perturbed_guess

        # Step 2: Local optimization using BFGS
        while self.evaluations < self.budget:
            # Adjust learning rate dynamically based on evaluations
            adaptive_options = {'maxiter': self.budget - self.evaluations, 'learning_rate': 0.05 * np.sqrt(1 - self.evaluations / self.budget)}  # Refined adaptive learning rate
            res = minimize(func, best_solution, bounds=bounds, method='L-BFGS-B', options=adaptive_options)
            self.evaluations += res.nfev
            if res.fun < best_score:
                best_score = res.fun
                best_solution = res.x
            else:  # Added dynamic restart mechanism
                restart_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
                scale_factor_restart = 0.05 * (self.budget - self.evaluations) / self.budget  # Adjusted perturbation for restart
                restart_guess += np.random.normal(0, scale_factor_restart, self.dim)
                res = minimize(func, restart_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget - self.evaluations, 'learning_rate': 0.1 * (1 - self.evaluations / self.budget)})  # Refined adaptive learning rate for restarts
                self.evaluations += res.nfev
                if res.fun < best_score:
                    best_score = res.fun
                    best_solution = res.x    
            if self.evaluations >= self.budget:
                break

        return best_solution