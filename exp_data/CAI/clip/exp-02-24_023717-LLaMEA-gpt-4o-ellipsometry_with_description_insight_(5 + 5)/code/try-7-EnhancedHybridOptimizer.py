import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        
        # Calculate the number of initial samples based on budget
        # Reserve half the budget for L-BFGS-B optimization
        num_initial_samples = max(1, self.budget // 2)
        
        # Uniform random sampling for initial guesses
        initial_solutions = np.random.uniform(low=lb, high=ub, size=(num_initial_samples, self.dim))
        
        # Apply Gaussian mutation to initial guesses for diversity
        mutation_strength = 0.05 * (np.array(ub) - np.array(lb))
        mutated_solutions = initial_solutions + np.random.normal(scale=mutation_strength, size=initial_solutions.shape)
        
        # Ensure mutated solutions are within bounds
        mutated_solutions = np.clip(mutated_solutions, lb, ub)
        
        best_solution = None
        best_value = np.inf
        
        # Evaluate initial and mutated solutions
        evaluations = 0
        all_initial_solutions = np.vstack((initial_solutions, mutated_solutions))
        for solution in all_initial_solutions:
            if evaluations >= self.budget:
                break
            value = func(solution)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = solution
        
        # L-BFGS-B optimization from the best initial solution
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