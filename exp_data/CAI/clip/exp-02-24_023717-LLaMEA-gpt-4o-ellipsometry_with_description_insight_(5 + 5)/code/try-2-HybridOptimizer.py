import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        
        # Calculate the number of initial samples based on budget
        # Reserve half the budget for BFGS optimization
        num_initial_samples = max(1, self.budget // 2)
        
        # Uniform random sampling for initial guesses
        initial_solutions = np.random.uniform(low=lb, high=ub, size=(num_initial_samples, self.dim))
        best_solution = None
        best_value = np.inf
        
        # Evaluate initial solutions
        evaluations = 0
        for i in range(num_initial_samples):
            if evaluations >= self.budget:
                break
            value = func(initial_solutions[i])
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = initial_solutions[i]
        
        # BFGS optimization from the best initial solution
        def local_objective(x):
            nonlocal evaluations
            if evaluations < self.budget:
                evaluations += 1
                return func(x)
            else:
                return np.inf
        
        if evaluations < self.budget:
            result = minimize(local_objective, best_solution, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        return best_solution, best_value