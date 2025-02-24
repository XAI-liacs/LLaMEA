import numpy as np

class StochasticGradientDescentWithAdaptiveSampling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.learning_rate = 0.1
    
    def __call__(self, func):
        remaining_budget = self.budget
        
        # Initial random guess within bounds
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        best_solution = current_solution.copy()
        best_value = func(current_solution)
        remaining_budget -= 1
        
        # Initialize learning rate and sampling range
        sampling_range = np.array([func.bounds.ub[i] - func.bounds.lb[i] for i in range(self.dim)])
        
        while remaining_budget > 0:
            # Generate a new candidate solution using SGD with adaptive sampling
            gradient = self.estimate_gradient(func, current_solution, remaining_budget)
            new_solution = current_solution - self.learning_rate * gradient
            
            # Clamp solution within bounds
            new_solution = np.clip(new_solution, func.bounds.lb, func.bounds.ub)
            
            # Evaluate and update if we found a better solution
            new_value = func(new_solution)
            remaining_budget -= 1
            
            if new_value < best_value:
                best_value = new_value
                best_solution = new_solution.copy()
                # Increase learning rate for faster convergence
                self.learning_rate *= 1.1
            else:
                # Reduce learning rate to refine search
                self.learning_rate *= 0.9
            
            current_solution = new_solution
        
        return best_solution
    
    def estimate_gradient(self, func, solution, remaining_budget):
        # Estimate gradient using finite differences
        gradient = np.zeros(self.dim)
        epsilon = 1e-5
        
        for i in range(self.dim):
            if remaining_budget <= 0:
                break
            
            direction = np.zeros(self.dim)
            direction[i] = epsilon
            
            plus_epsilon = np.clip(solution + direction, func.bounds.lb, func.bounds.ub)
            minus_epsilon = np.clip(solution - direction, func.bounds.lb, func.bounds.ub)
            
            gradient[i] = (func(plus_epsilon) - func(minus_epsilon)) / (2 * epsilon)
            remaining_budget -= 2
        
        return gradient