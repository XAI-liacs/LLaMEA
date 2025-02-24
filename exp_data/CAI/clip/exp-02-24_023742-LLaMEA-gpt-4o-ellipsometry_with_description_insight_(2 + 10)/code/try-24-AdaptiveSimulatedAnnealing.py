import numpy as np

class AdaptiveSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        def simulated_annealing(func, x0, bounds, max_evals):
            current_x = np.clip(x0, bounds.lb, bounds.ub)
            current_f = func(current_x)
            best_x, best_f = current_x, current_f
            
            temp = 1.0
            cooling_rate = 0.95
            
            for i in range(max_evals):
                if self.evals >= self.budget:
                    break
                
                # Generate a new candidate solution
                candidate_x = current_x + np.random.uniform(-0.1, 0.1, self.dim) * (bounds.ub - bounds.lb)
                candidate_x = np.clip(candidate_x, bounds.lb, bounds.ub)
                
                candidate_f = func(candidate_x)
                self.evals += 1
                
                # Determine if the candidate is accepted
                if candidate_f < current_f or np.random.rand() < np.exp((current_f - candidate_f) / temp):
                    current_x, current_f = candidate_x, candidate_f
                    if current_f < best_f:
                        best_x, best_f = current_x, current_f
                
                # Update the temperature
                temp *= cooling_rate
            
            return best_x, best_f
        
        # Multi-start approach
        remaining_budget = self.budget
        best_x, best_f = None, float('inf')
        
        while remaining_budget > 0:
            # Randomly initialize a starting point
            x0 = np.random.uniform(lb, ub, self.dim)
            max_evals = min(remaining_budget, 100)
            
            x_opt, f_opt = simulated_annealing(func, x0, func.bounds, max_evals)
            
            if f_opt < best_f:
                best_x, best_f = x_opt, f_opt
            
            remaining_budget -= max_evals
        
        return best_x