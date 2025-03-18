import numpy as np
from scipy.optimize import minimize

class AdaptiveBiPhaseOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        num_initial_guesses = min(self.budget // 20, 10)
        
        # Generate initial guesses using uniform sampling across bounds
        initial_guesses = np.array([np.random.uniform(low=bound[0], high=bound[1], size=self.dim) for bound in [bounds] * num_initial_guesses])
        
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        epsilon = 1e-6  # Convergence threshold
        
        while evaluations < self.budget:
            for guess in initial_guesses:
                if evaluations >= self.budget:
                    break
                
                # Local exploitation using L-BFGS-B with dynamic max function calls
                local_budget = min(self.budget - evaluations, 50)
                result = minimize(func, guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': local_budget, 'ftol': epsilon})
                evaluations += result.nfev
                
                # Update best known solution
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
                    if best_value < epsilon:  # Early stopping condition
                        return best_solution

            # Global exploration: reset some guesses based on a fraction of remaining budget
            exploration_factor = (self.budget - evaluations) / self.budget
            if exploration_factor > 0.5:  # High exploration phase
                new_guesses = np.array([np.random.uniform(low=bound[0], high=bound[1], size=self.dim) for bound in [bounds] * num_initial_guesses])
                initial_guesses = new_guesses

        return best_solution