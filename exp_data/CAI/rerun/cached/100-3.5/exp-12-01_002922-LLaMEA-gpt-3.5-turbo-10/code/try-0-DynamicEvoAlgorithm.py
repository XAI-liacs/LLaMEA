import numpy as np

class DynamicEvoAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 2 * dim
        self.mutation_rate = 0.5
        self.mutation_step = 0.1

    def __call__(self, func):
        best_solution = 10.0 * np.random.rand(self.dim) - 5.0
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // self.pop_size):
            population = [10.0 * np.random.rand(self.dim) - 5.0 for _ in range(self.pop_size)]
            for individual in population:
                mutated_individual = individual + self.mutation_step * np.random.randn(self.dim)
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
                fitness = func(mutated_individual)
                if fitness < best_fitness:
                    best_solution = mutated_individual
                    best_fitness = fitness
        return best_solution