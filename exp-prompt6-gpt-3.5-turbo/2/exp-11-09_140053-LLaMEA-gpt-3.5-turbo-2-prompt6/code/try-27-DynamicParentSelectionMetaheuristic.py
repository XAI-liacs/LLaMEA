import numpy as np

class DynamicParentSelectionMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_rate = np.full(self.pop_size, 0.1)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget):
            offspring = []
            for i in range(self.pop_size):
                diversity = np.std(population, axis=0)
                parent_indices = np.random.choice(range(self.pop_size), 2, replace=False, p=diversity / np.sum(diversity))
                parent1, parent2 = population[parent_indices]
                child = parent1 + self.mutation_rate[i] * np.random.randn(self.dim) + self.mutation_rate[i] * np.random.randn(self.dim)
                if func(child) < min(fitness[parent_indices]):
                    population[i] = child
                    fitness[i] = func(child)
                    self.mutation_rate[i] *= 1.02 + 0.005 * np.mean(self.mutation_rate)  # Dynamic mutation rate update
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))  # Dynamic population size adaptation
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))  # Add new individuals
                else:
                    self.mutation_rate[i] *= 0.98 - 0.005 * np.mean(self.mutation_rate)  # Dynamic mutation rate update
                    self.pop_size = int(10 * (1 - np.mean(self.mutation_rate)))  # Dynamic population size adaptation
                    population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim)))  # Add new individuals
        return population[np.argmin(fitness)]