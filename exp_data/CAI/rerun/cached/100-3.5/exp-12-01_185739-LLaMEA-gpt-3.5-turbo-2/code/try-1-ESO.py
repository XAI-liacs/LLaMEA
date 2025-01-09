import numpy as np

class ESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.num_iterations = budget // self.population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        for _ in range(self.num_iterations):
            fitness = [func(individual) for individual in population]
            sorted_indices = np.argsort(fitness)
            elite = population[sorted_indices[0]]

            for i in range(self.population_size):
                if i != sorted_indices[0]:
                    population[i] = np.clip(elite + np.random.normal(0, 1, self.dim), self.lower_bound, self.upper_bound)

        return elite