import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize bounds 
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        
        # Multiple initial guesses to enhance initial exploration
        num_initial_guesses = 3
        initial_guesses = [np.random.uniform(bounds[0], bounds[1], self.dim) for _ in range(num_initial_guesses)]
        
        # Evaluate initial guesses and sort by performance
        evaluated_guesses = [(guess, func(guess)) for guess in initial_guesses]
        evaluated_guesses.sort(key=lambda x: x[1])
        
        # Introduce perturbed elite solutions for initial guesses
        best_guess = evaluated_guesses[0][0]
        perturbed_guesses = [best_guess + np.random.uniform(-0.05, 0.05, self.dim) for _ in range(num_initial_guesses)]
        
        # Define the optimization callback to count function evaluations
        self.func_evaluations = 0
        def callback(x):
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                raise StopIteration("Budget exhausted")

        try:
            result = None
            # Use the best initial guess based on initial evaluations
            for initial_guess, _ in zip(perturbed_guesses, evaluated_guesses):
                try:
                    result = minimize(func, initial_guess, method='Nelder-Mead', 
                                      bounds=bounds.T, callback=callback, 
                                      options={'maxiter': self.budget, 'disp': False})
                except StopIteration:
                    break

                # Adjust bounds based on the found solution for potential further refinement
                if result.success:
                    center = result.x
                    bounds = np.clip(bounds, func.bounds.lb, func.bounds.ub)
                    # Introduce a dynamic adjustment factor for bounds refinement
                    adjust_factor = 0.15 * (1 + np.random.uniform(-0.05, 0.05))
                    new_bounds = np.array([np.maximum(center - adjust_factor * (bounds[1] - bounds[0]), bounds[0]),
                                           np.minimum(center + adjust_factor * (bounds[1] - bounds[0]), bounds[1])])
                    
                    # Re-run optimization with updated bounds if budget allows
                    try:
                        result = minimize(func, result.x, method='Nelder-Mead', 
                                          bounds=new_bounds.T, callback=callback, 
                                          options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
                    except StopIteration:
                        break
            
        except StopIteration:
            pass

        return result.x if result else initial_guesses[0]  # Return initial if no result was obtained