import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.learning_rate = 0.1  # Adaptive learning rate

    def __call__(self, func):
        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Uniform sampling for initial guesses, dynamically adjusted based on remaining budget
        num_initial_samples = max(5, (self.budget - self.evaluations) // 6)
        samples = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        best_solution = None
        best_value = float('inf')
        
        # Begin search
        for sample in samples:
            # Local optimization using L-BFGS-B
            res = minimize(func, sample, bounds=list(zip(lb, ub)), method='L-BFGS-B', options={'ftol': 1e-9})
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
            
            if self.evaluations >= self.budget:
                break

        # Iteratively adjust bounds and refine
        while self.evaluations < self.budget:
            # Narrow the bounds based on current best solution
            new_lb = np.maximum(lb, best_solution - self.learning_rate * (ub - lb))
            new_ub = np.minimum(ub, best_solution + self.learning_rate * (ub - lb))
            
            # Local optimization within adjusted bounds
            res = minimize(func, best_solution, bounds=list(zip(new_lb, new_ub)), method='L-BFGS-B', options={'ftol': 1e-9})
            self.evaluations += res.nfev
            
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
                self.learning_rate *= 0.9  # Decrease learning rate for convergence
                
            else:
                self.learning_rate *= 1.1  # Increase learning rate if no improvement
                
            # Randomized restart strategy
            if self.evaluations < self.budget and np.random.rand() < 0.1:
                restart_sample = np.random.uniform(lb, ub, self.dim)
                res = minimize(func, restart_sample, bounds=list(zip(lb, ub)), method='L-BFGS-B', options={'ftol': 1e-9})
                self.evaluations += res.nfev
                
                if res.fun < best_value:
                    best_value = res.fun
                    best_solution = res.x
            
            # Stopping criterion if convergence is achieved
            if abs(res.fun - best_value) < 1e-6:
                break

        return best_solution