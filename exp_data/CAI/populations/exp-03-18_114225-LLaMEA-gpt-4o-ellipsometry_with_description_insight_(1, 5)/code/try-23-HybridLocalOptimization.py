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
        
        # Step 1: Enhanced uniform random sampling with small re-evaluation
        initial_samples = min(self.budget // 6, 10)  # Adjusted initial sample size to reduce sample variance
        for _ in range(initial_samples):
            initial_guess = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            perturbed_guess = initial_guess + np.random.normal(0, 0.01 + (self.budget - self.evaluations) / (8 * self.budget), self.dim)  # Adjusted dynamic perturbation
            score = func(perturbed_guess)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = perturbed_guess
            # Re-evaluate the best-found guess to exploit local search more
            re_score = func(best_solution)
            self.evaluations += 1
            if re_score < best_score:
                best_score = re_score

        # Step 2: Local optimization using BFGS
        while self.evaluations < self.budget:
            res = minimize(func, best_solution, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget - self.evaluations})
            self.evaluations += res.nfev
            if res.fun < best_score:
                best_score = res.fun
                best_solution = res.x
            if self.evaluations >= self.budget:
                break

        return best_solution