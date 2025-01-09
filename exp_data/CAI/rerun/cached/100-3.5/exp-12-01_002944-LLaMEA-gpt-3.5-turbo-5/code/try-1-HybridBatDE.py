import numpy as np

class HybridBatDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.loudness = 0.5
        self.pulse_rate = 0.5
        self.alpha = 0.9
        self.gamma = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_solution = population[np.argmin(fitness)]
        for _ in range(self.budget):
            new_population = population.copy()
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    frequency = np.mean(fitness)
                    new_population[i] += (best_solution - population[i]) * self.loudness
                    new_population[i] = np.clip(new_population[i], self.lower_bound, self.upper_bound)
                    if np.random.rand() < self.alpha:
                        best_bat = population[np.argmin(fitness)]
                        new_population[i] += self.gamma * (best_bat - new_population[i])
            new_fitness = np.array([func(individual) for individual in new_population])
            population = new_population
            fitness = new_fitness
            best_solution = population[np.argmin(fitness)]
        return best_solution