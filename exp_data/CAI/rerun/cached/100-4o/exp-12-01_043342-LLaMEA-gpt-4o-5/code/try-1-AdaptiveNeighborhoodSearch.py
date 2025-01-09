import numpy as np

class AdaptiveNeighborhoodSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.history = []

    def __call__(self, func):
        current_budget = 0
        
        # Initialize a random solution
        solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(solution)
        self.history.append((solution, best_value))
        current_budget += 1
        
        # Adaptive neighborhood search parameters
        scale_factor = 0.5
        reduction_factor = 0.9
        
        while current_budget < self.budget:
            # Generate a candidate solution with a normal distribution
            candidate = solution + np.random.normal(0, scale_factor, self.dim)
            
            # Clip candidate to ensure it remains within bounds
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            
            # Evaluate the candidate solution
            candidate_value = func(candidate)
            current_budget += 1
            
            # If candidate is better, update the solution
            if candidate_value < best_value:
                solution = candidate
                best_value = candidate_value
                scale_factor = max(0.01, scale_factor * reduction_factor)  # Reduce scale_factor to refine search
                self.history.append((solution, best_value))
            else:
                scale_factor = min(1.0, scale_factor / reduction_factor)  # Increase scale_factor to explore more

        return solution, best_value