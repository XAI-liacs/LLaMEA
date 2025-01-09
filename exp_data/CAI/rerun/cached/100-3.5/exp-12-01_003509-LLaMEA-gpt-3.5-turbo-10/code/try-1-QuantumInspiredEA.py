import numpy as np
from scipy.stats import ortho_group

class QuantumInspiredEA:
    def __init__(self, budget, dim, population_size=50, num_generations=100, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]
        
        for _ in range(self.num_generations):
            rotation_matrix = ortho_group.rvs(self.dim)
            rotated_population = population @ rotation_matrix.T
            mutated_population = rotated_population + np.random.uniform(-1, 1, size=(self.population_size, self.dim)) * self.mutation_rate
            new_population = np.clip(mutated_population @ rotation_matrix, -5.0, 5.0)
            
            for individual in new_population:
                if self.budget == 0:
                    break
                fitness = func(individual)
                if fitness < func(best_solution):
                    best_solution = individual
                self.budget -= 1
                
        return best_solution