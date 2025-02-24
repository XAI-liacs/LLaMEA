import numpy as np
from scipy.optimize import minimize

class EAMSLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        num_initial_guesses = min(5, self.budget // 10)
        remaining_budget = self.budget
        
        initial_guesses = [
            np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            for _ in range(num_initial_guesses)
        ]
        
        best_solution = None
        best_value = float('inf')
        adaptive_rates = np.linspace(0.05, 0.2, num_initial_guesses)  # Adaptive learning rates

        for idx, init_guess in enumerate(initial_guesses):
            if remaining_budget <= 0:
                break
            
            bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
            
            result = minimize(
                func,
                init_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': min(remaining_budget, 10)}
            )
            
            remaining_budget -= result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                
            learning_rate = adaptive_rates[idx]
            for i in range(self.dim):
                bounds[i] = (
                    max(bounds[i][0], best_solution[i] - (func.bounds.ub[i] - func.bounds.lb[i]) * learning_rate),
                    min(bounds[i][1], best_solution[i] + (func.bounds.ub[i] - func.bounds.lb[i]) * learning_rate)
                )
        
        return best_solution