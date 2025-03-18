import numpy as np
from scipy.optimize import minimize

class SimulatedAnnealingGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def _acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost < old_cost:
            return 1.0
        else:
            return np.exp((old_cost - new_cost) / temperature)
        
    def __call__(self, func):
        bounds = np.array([(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
        
        # Initialize solution
        current_solution = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dim)
        current_cost = func(current_solution)
        self.budget -= 1
        
        best_solution = np.copy(current_solution)
        best_cost = current_cost
        
        temperature = 1.0
        cooling_rate = 0.95
        
        while self.budget > 0:
            # Generate new candidate solution
            candidate_solution = current_solution + np.random.normal(0, 0.1, self.dim)
            candidate_solution = np.clip(candidate_solution, bounds[:, 0], bounds[:, 1])
            candidate_cost = func(candidate_solution)
            self.budget -= 1
            
            # Decide acceptance
            if self._acceptance_probability(current_cost, candidate_cost, temperature) > np.random.rand():
                current_solution, current_cost = candidate_solution, candidate_cost
                
                # Update best solution
                if current_cost < best_cost:
                    best_solution, best_cost = current_solution, current_cost
            
            # Anneal
            temperature *= cooling_rate
            
            # Optional gradient-based refinement
            if self.budget > 0 and np.random.rand() < 0.2: # 20% chance to perform gradient-based refinement
                result = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': max(self.budget, 1)})
                if result.fun < best_cost:
                    best_solution, best_cost = result.x, result.fun
                    self.budget -= result.nfev
        
        return best_solution