import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        temperature = 1.0
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            parents = self.population[np.argsort(fitness)[:2]]
            offspring = 0.5 * (parents[0] + parents[1]) + np.random.normal(0, 1, self.dim)
            worst_idx = np.argmax(fitness)
            if np.random.rand() < np.exp((fitness[worst_idx] - func(offspring)) / temperature):
                self.population[worst_idx] = offspring
            temperature *= 0.9  # Cooling schedule
        return self.population[np.argmin([func(individual) for individual in self.population])]