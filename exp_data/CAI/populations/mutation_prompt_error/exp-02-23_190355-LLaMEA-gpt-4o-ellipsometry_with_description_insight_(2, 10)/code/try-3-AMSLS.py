import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AMSLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        num_initial_guesses = min(5, self.budget // 10)  # Limit the number of initial guesses
        remaining_budget = self.budget
        
        # Generate diverse initial guesses across the search space using Sobol sequences
        sobol = Sobol(d=self.dim, scramble=True)
        initial_guesses = sobol.random_base2(m=int(np.log2(num_initial_guesses)))
        initial_guesses = initial_guesses * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
        
        best_solution = None
        best_value = float('inf')

        # Iteratively optimize from different starting points
        for init_guess in initial_guesses:
            if remaining_budget <= 0:
                break
            
            bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
            
            result = minimize(
                func,
                init_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': min(remaining_budget, 10)}  # Restrict function evaluations per run
            )
            
            remaining_budget -= result.nfev
            
            # Update best known solution
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
                
            # Adapt search bounds based on current best solution
            for i in range(self.dim):
                bounds[i] = (
                    max(bounds[i][0], best_solution[i] - (func.bounds.ub[i] - func.bounds.lb[i]) * 0.1),
                    min(bounds[i][1], best_solution[i] + (func.bounds.ub[i] - func.bounds.lb[i]) * 0.1)
                )
        
        return best_solution