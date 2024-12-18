import numpy as np

class FireAntOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        
        chaos_value = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)  # Introducing chaotic initialization
        
        for _ in range(self.budget):
            new_solution = best_solution + np.random.uniform(-1, 1, self.dim) + chaos_value  # Adding chaos for exploration
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
        
        return best_solution