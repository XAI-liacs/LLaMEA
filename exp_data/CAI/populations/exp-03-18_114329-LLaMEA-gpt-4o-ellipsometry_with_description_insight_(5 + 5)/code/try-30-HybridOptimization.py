import numpy as np
from scipy.optimize import minimize

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        num_initial_guesses = min(10, self.budget // 2)
        best_solution = None
        best_value = float('inf')
        
        lb, ub = func.bounds.lb, func.bounds.ub
        
        initial_guesses = np.random.uniform(lb, ub, (num_initial_guesses, self.dim))
        evaluations = 0
        early_stop_threshold = 1e-8  # Changed line for improved early stopping criterion

        for initial_guess in initial_guesses:
            if evaluations >= self.budget:
                break
            
            res = minimize(func, initial_guess, method='L-BFGS-B', bounds=zip(lb, ub))
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
                if best_value < early_stop_threshold:
                    break

            if evaluations < self.budget:
                factor = 0.4  # Changed line for refined exploration factor
                new_bounds = [(max(lb[i], res.x[i] - factor * (ub[i] - lb[i])), 
                               min(ub[i], res.x[i] + factor * (ub[i] - lb[i]))) for i in range(self.dim)]
                for _ in range(3):
                    if evaluations >= self.budget:
                        break
                    res = minimize(func, res.x, method='L-BFGS-B', bounds=new_bounds)
                    evaluations += res.nfev
                    if res.fun < best_value:
                        best_value = res.fun
                        best_solution = res.x
                        
        return best_solution