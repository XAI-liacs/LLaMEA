import numpy as np

class QIEA:
    def __init__(self, budget, dim, pop_size=50, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        def evolve_population(population, fitness):
            fitness_order = np.argsort(fitness)
            new_population = []
            for i in range(self.pop_size):
                parent1 = population[fitness_order[i % self.pop_size]]
                parent2 = population[fitness_order[(i+1) % self.pop_size]]
                child = parent1 + np.random.uniform(-1, 1, self.dim) * (parent2 - parent1)
                if np.random.rand() < self.mutation_rate:
                    child += np.random.normal(0, 1, self.dim)
                new_population.append(child)
            return np.array(new_population)
        
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget // self.pop_size):
            new_population = evolve_population(population, fitness)
            new_fitness = np.array([func(individual) for individual in new_population])
            mask = new_fitness < fitness
            population[mask] = new_population[mask]
            fitness[mask] = new_fitness[mask]
            
        best_idx = np.argmin(fitness)
        return population[best_idx]