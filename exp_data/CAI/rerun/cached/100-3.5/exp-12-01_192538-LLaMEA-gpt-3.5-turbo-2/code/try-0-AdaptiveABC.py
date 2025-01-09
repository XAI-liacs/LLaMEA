import numpy as np

class AdaptiveABC:
    def __init__(self, budget, dim, colony_size=50):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        best_solution = np.random.uniform(lower_bound, upper_bound, size=self.dim)
        best_fitness = func(best_solution)
        
        colony = np.random.uniform(low=lower_bound, high=upper_bound, size=(self.colony_size, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.colony_size):
                new_solution = colony[i] + np.random.uniform(-1, 1, size=self.dim) * (colony[np.random.choice(np.setdiff1d(range(self.colony_size), i))] - colony[np.random.choice(np.setdiff1d(range(self.colony_size), i))])
                new_solution = np.clip(new_solution, lower_bound, upper_bound)
                new_fitness = func(new_solution)
                
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                
                if np.random.uniform() < 0.5:
                    colony[i] = new_solution
        
        return best_solution