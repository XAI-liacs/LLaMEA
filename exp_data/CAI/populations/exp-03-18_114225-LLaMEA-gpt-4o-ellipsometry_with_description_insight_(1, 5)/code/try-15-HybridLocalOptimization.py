import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridLocalOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        best_solution = None
        best_score = float('inf')
        
        # Step 1: Sobol sequence sampling for initial guesses
        initial_samples = min(self.budget // 8, 10)  # Limited to 12.5% of budget or 10 samples
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random(n=initial_samples)
        scaled_samples = np.array([func.bounds.lb + sample * (func.bounds.ub - func.bounds.lb) for sample in samples])
        for initial_guess in scaled_samples:
            score = func(initial_guess)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = initial_guess

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