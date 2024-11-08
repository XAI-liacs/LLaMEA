import numpy as np

class EnhancedDEAdaptiveImprovedOptimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f = 0.5
        self.cr = 0.9
        self.population_size = 10
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])

        for _ in range(self.budget - self.population_size):
            mutants_idx = np.random.choice(self.population_size, (self.population_size, 2), replace=True)
            crossover = np.random.rand(self.population_size, self.dim) < self.cr
            mutants = population[mutants_idx]

            mutants_diff = self.f * (mutants[:, 0] - mutants[:, 1])
            population += crossover * np.where(crossover, np.clip(mutants_diff, -self.f, self.f), 0)

            new_fitness_values = np.array([func(ind) for ind in population])
            improved_indices = new_fitness_values < fitness_values
            fitness_values[improved_indices] = new_fitness_values[improved_indices]

        best_index = np.argmin(fitness_values)
        return population[best_index]