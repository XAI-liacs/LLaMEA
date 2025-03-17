import numpy as np

class AdaptiveSpiralOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        center = (lb + ub) / 2.0  # Start from the center of the search space
        radius = (ub - lb) / 2.0  # Initial search radius

        best_solution = center
        best_value = func(center)
        
        evaluations = 1
        while evaluations < self.budget:
            for angle in np.linspace(0, 2*np.pi, self.dim, endpoint=False):
                direction = np.array([np.cos(angle), np.sin(angle)])
                step_size = radius * (1 - (evaluations / self.budget))
                
                candidate_solution = best_solution + step_size * direction
                candidate_solution = np.clip(candidate_solution, lb, ub)
                
                candidate_value = func(candidate_solution)
                evaluations += 1

                if candidate_value < best_value:
                    best_solution = candidate_solution
                    best_value = candidate_value

                if evaluations >= self.budget:
                    break

            radius *= 0.9  # Reduce the radius over time to focus on local search

        return best_solution

# To use this class, instantiate it with the budget and dimension, then call it with the function to optimize.