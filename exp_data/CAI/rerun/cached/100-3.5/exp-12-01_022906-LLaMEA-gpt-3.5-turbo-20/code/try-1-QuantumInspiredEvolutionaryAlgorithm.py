import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = [func(individual) for individual in self.population]
            sorted_indices = np.argsort(fitness_values)
            elite = self.population[sorted_indices[0]]
            for i in range(1, self.budget):
                self.population[i] = np.clip(elite + np.random.uniform(-1, 1, self.dim), -5.0, 5.0)
        best_index = np.argmin([func(individual) for individual in self.population])
        return self.population[best_index]