import numpy as np

class PercolationAlgorithm:
    def __init__(self, budget, dim, pop_size=50, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget // self.pop_size):
            sorted_indices = np.argsort(fitness)
            elite_pop = population[sorted_indices[:self.pop_size // 2]]
            non_elite_pop = population[sorted_indices[self.pop_size // 2:]]
            
            for i in range(self.pop_size // 2, self.pop_size):
                parent_idx = np.random.choice(range(self.pop_size // 2))
                mutation_mask = np.random.choice([0, 1], size=self.dim, p=[1-self.mutation_rate, self.mutation_rate])
                population[i] = elite_pop[parent_idx] + mutation_mask * np.random.uniform(-1.0, 1.0, self.dim)
                fitness[i] = func(population[i])
        
        best_idx = np.argmin(fitness)
        return population[best_idx]