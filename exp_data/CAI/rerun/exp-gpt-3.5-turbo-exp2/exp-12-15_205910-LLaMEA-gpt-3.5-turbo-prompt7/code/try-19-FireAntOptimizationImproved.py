import numpy as np

class FireAntOptimizationImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.step_size = 1.0

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            step = self.step_size * np.random.uniform(-1, 1, self.dim)
            new_solution = best_solution + step
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                self.step_size *= 0.99  # Dynamic step size adaptation
        
        return best_solution