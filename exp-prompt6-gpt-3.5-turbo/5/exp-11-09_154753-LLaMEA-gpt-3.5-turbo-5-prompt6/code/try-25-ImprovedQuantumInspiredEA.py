import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            parents = self.population[np.argsort(fitness)[:2]]
            
            # Introduce dynamic mutation based on individual fitness
            mutation_scale = np.maximum(0.1, 1 / (1 + np.exp(-2 * np.array(fitness))))
            offspring = 0.5 * (parents[0] + parents[1]) + mutation_scale * np.random.normal(0, 1, self.dim)
            
            worst_idx = np.argmax(fitness)
            self.population[worst_idx] = offspring
        return self.population[np.argmin([func(individual) for individual in self.population])]