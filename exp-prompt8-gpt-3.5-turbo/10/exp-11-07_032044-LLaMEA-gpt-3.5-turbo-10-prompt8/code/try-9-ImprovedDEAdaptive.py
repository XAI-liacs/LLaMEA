import numpy as np

class ImprovedDEAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f = 0.5
        self.cr = 0.9
        self.population_size = 10
        self.bounds = (-5.0, 5.0)
    
    def __call__(self, func):
        def clip_to_bounds(x):
            return np.clip(x, *self.bounds)
        
        population = np.random.uniform(*self.bounds, (self.population_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            mutant_pop = population + self.f * (population[np.random.choice(np.arange(self.population_size), (self.population_size, 1), replace=True)] - population[np.random.choice(np.arange(self.population_size), (self.population_size, 1), replace=True)])
            crossover = np.random.rand(self.population_size, self.dim) < self.cr
            new_population = np.where(crossover, clip_to_bounds(mutant_pop), population)
            
            new_fitness_values = np.array([func(ind) for ind in new_population])
            improved_indices = new_fitness_values < fitness_values
            population[improved_indices] = new_population[improved_indices]
            fitness_values[improved_indices] = new_fitness_values[improved_indices]
        
        best_index = np.argmin(fitness_values)
        return population[best_index]