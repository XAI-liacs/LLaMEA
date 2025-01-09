import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.num_iterations = budget // self.population_size
        self.alpha = 0.5
        self.beta = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for _ in range(self.num_iterations):
            offspring = np.array([np.clip(individual + np.random.normal(0, 1, self.dim), -5.0, 5.0) for individual in population])
            fitness_values = np.array([func(individual) for individual in offspring])
            sorted_indices = np.argsort(fitness_values)
            population = (1 - self.alpha) * population + self.alpha * offspring[sorted_indices[:self.population_size]]
            population += self.beta * np.random.normal(0, 1, (self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution