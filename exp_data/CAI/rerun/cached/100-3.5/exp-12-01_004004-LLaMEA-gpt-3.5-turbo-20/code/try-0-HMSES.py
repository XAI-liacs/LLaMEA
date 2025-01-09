import numpy as np

class HMSES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.strategies = [1, 1.5, 2]  # Different mutation strategies
        self.population_size = 10  # Number of individuals in the population

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        
        for _ in range(self.budget // self.population_size):
            for strategy in self.strategies:
                population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
                for i in range(self.population_size):
                    mutated_solution = population[i] + np.random.normal(0, strategy, self.dim)
                    mutated_solution = np.clip(mutated_solution, -5.0, 5.0)
                    fitness = func(mutated_solution)
                    
                    if fitness < best_fitness:
                        best_solution = mutated_solution
                        best_fitness = fitness
                        
        return best_solution