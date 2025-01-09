import numpy as np

class DynamicEvoAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.crossover_rate = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness_values = [func(individual) for individual in population]
        
        for _ in range(self.budget // self.population_size):
            new_population = []
            for i in range(self.population_size):
                idxs = np.random.choice(range(self.population_size), 3, replace=False)
                mutant = population[idxs[0]] + 0.5 * (population[idxs[1]] - population[idxs[2]])
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                offspring = np.where(crossover_mask, mutant, population[i])
                new_population.append(offspring)
            
            new_fitness_values = [func(individual) for individual in new_population]
            combined_population = list(zip(population, fitness_values)) + list(zip(new_population, new_fitness_values))
            combined_population.sort(key=lambda x: x[1])
            population = [ind for ind, _ in combined_population[:self.population_size]]
            fitness_values = [fit for _, fit in combined_population[:self.population_size]]

        return population[np.argmin(fitness_values)]