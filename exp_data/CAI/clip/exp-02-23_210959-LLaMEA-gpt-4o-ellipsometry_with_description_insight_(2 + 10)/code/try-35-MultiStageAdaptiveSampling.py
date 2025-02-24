import numpy as np
from scipy.optimize import minimize

class MultiStageAdaptiveSampling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Determine bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        remaining_budget = self.budget

        # Stage 1: Coarse uniform sampling to get an overview of the landscape
        stage1_samples = max(5, self.budget // 20)
        initial_guesses = np.random.uniform(lb, ub, size=(stage1_samples, self.dim))
        
        # Evaluate initial guesses
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        for guess in initial_guesses:
            if evaluations >= self.budget:
                break
            value = func(guess)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = guess

        # Stage 2: Refine the search using narrowed bounds around the best solution
        stage2_samples = min(20, self.budget // 10)  # Adjusted sample count
        refined_lb = np.maximum(lb, best_solution - (ub - lb) * 0.1)
        refined_ub = np.minimum(ub, best_solution + (ub - lb) * 0.1)
        refined_guesses = np.random.uniform(refined_lb, refined_ub, size=(stage2_samples, self.dim))
        
        for guess in refined_guesses:
            if evaluations >= self.budget:
                break
            value = func(guess)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = guess
        
        # Stage 3: Use Gradient Descent for local optimization
        def wrapped_func(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return float('inf')
            value = func(x)
            evaluations += 1
            return value

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'ftol': 1e-9})

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        return best_solution