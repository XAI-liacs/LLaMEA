import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
    
    def quantum_rotation_gate(self, x):
        return x * np.exp(1j * np.pi / 2)
    
    def quantum_crossover(self, parent1, parent2):
        return (parent1 + parent2) / 2
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = [func(individual) for individual in self.population]
            sorted_indices = np.argsort(fitness_values)
            elite = self.population[sorted_indices[0]]
            for i in range(1, self.population_size):
                j = np.random.choice(np.delete(sorted_indices, i))
                crossovered = self.quantum_crossover(self.population[i], self.population[j])
                self.population[i] = self.quantum_rotation_gate(crossovered)
        return elite