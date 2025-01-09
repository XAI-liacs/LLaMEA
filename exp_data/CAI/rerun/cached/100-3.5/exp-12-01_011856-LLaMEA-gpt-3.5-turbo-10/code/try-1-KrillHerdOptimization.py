import numpy as np

class KrillHerdOptimization:
    def __init__(self, budget, dim, population_size=50, step_size=0.1, c1=0.1, c2=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.step_size = step_size
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            # Update the position of krill based on attraction and repulsion
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        population[i] += self.step_size * (population[j] - population[i]) * self.c1
                    else:
                        population[i] -= self.step_size * (population[j] - population[i]) * self.c2
            
            # Update fitness values
            fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution