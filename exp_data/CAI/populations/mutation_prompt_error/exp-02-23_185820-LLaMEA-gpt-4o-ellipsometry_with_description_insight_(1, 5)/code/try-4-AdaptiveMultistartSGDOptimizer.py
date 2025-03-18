import numpy as np
from scipy.optimize import minimize

class AdaptiveMultistartSGDOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        bounds = np.array(list(zip(func.bounds.lb, func.bounds.ub)))
        num_initial_samples = min(10, self.budget // 4)
        
        # Step 1: Uniform sampling for diverse initial guesses
        initial_samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_initial_samples, self.dim))
        initial_values = [func(x) for x in initial_samples]
        self.evals += num_initial_samples
        
        # Step 2: Multi-start local search with gradient-based refinement
        best_solution = None
        best_value = float('inf')
        
        for i in range(num_initial_samples):
            guess = initial_samples[i]
            
            # Local optimizer using L-BFGS-B
            result = minimize(
                func,
                guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': (self.budget - self.evals) // (num_initial_samples - i)}
            )
            self.evals += result.nfev
            
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Early exit if budget is exhausted
            if self.evals >= self.budget:
                break
        
        # Step 3: Stochastic Gradient Descent for final exploitation phase
        alpha = 0.01  # learning rate
        current_solution = best_solution
        current_value = best_value
        
        while self.evals < self.budget:
            gradient = self._estimate_gradient(func, current_solution)
            next_solution = current_solution - alpha * gradient
            
            # Clip to bounds
            next_solution = np.clip(next_solution, bounds[:, 0], bounds[:, 1])
            
            next_value = func(next_solution)
            self.evals += 1
            
            if next_value < current_value:
                current_solution = next_solution
                current_value = next_value
            
        return current_solution
    
    def _estimate_gradient(self, func, x):
        epsilon = 1e-8
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_step = np.array(x)
            x_step[i] += epsilon
            grad[i] = (func(x_step) - func(x)) / epsilon
        return grad