import numpy as np
from scipy.optimize import minimize, differential_evolution

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
        initial_samples = min(self.budget // 8, 10)
        for _ in range(initial_samples):
            initial_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            scale_factor = 0.01 * np.std(initial_guess) * (1 - self.evaluations / self.budget) * np.log1p(self.budget)
            perturbed_guess = initial_guess + np.random.normal(0, scale_factor, self.dim)
            score = func(perturbed_guess)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = perturbed_guess

        # Step 2: Local optimization using L-BFGS-B
        while self.evaluations < self.budget:
            if self.evaluations < self.budget * 0.5:  # Use differential evolution in the first half of budget
                adaptive_options = {'maxiter': self.budget - self.evaluations}
                res = differential_evolution(func, bounds, strategy='best1bin', maxiter=adaptive_options['maxiter'])
            else:  # Use local search in the latter half
                adaptive_options = {'maxiter': self.budget - self.evaluations}
                res = minimize(func, best_solution, bounds=bounds, method='L-BFGS-B', options=adaptive_options)
                
            self.evaluations += res.nfev if hasattr(res, 'nfev') else res.nit
            if res.fun < best_score:
                best_score = res.fun
                best_solution = res.x
            else:
                restart_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
                res = minimize(func, restart_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget - self.evaluations})
                self.evaluations += res.nfev
                if res.fun < best_score:
                    best_score = res.fun
                    best_solution = res.x    
            if self.evaluations >= self.budget:
                break

        return best_solution