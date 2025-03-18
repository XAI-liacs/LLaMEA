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
        
        # Step 1: Initial sampling with Differential Evolution to ensure diverse exploration
        def objective(x):
            nonlocal best_score, best_solution
            score = func(x)
            self.evaluations += 1
            if score < best_score:
                best_score = score
                best_solution = x
            return score

        initial_samples = min(self.budget // 10, 10)
        result = differential_evolution(objective, bounds, maxiter=initial_samples, popsize=5, polish=False)
        
        # Update best solution found
        if result.fun < best_score:
            best_score = result.fun
            best_solution = result.x

        # Step 2: Local optimization using BFGS after DE
        while self.evaluations < self.budget:
            res = minimize(func, best_solution, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.budget - self.evaluations})
            self.evaluations += res.nfev
            if res.fun < best_score:
                best_score = res.fun
                best_solution = res.x
            if self.evaluations >= self.budget:
                break

        return best_solution