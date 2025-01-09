import numpy as np

class FishSchoolSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.step_size = 0.2
        self.delta_d = 0.97
        self.delta_s = 0.9
        self.fitness = np.inf
        self.best_solution = None

    def __call__(self, func):
        self.initialize_population()
        evals = 0
        while evals < self.budget:
            for i in range(self.population_size):
                current_fitness = func(self.population[i])
                evals += 1

                if current_fitness < self.fitness:
                    self.fitness = current_fitness
                    self.best_solution = np.copy(self.population[i])

                step = self.step_size * np.random.uniform(-1, 1, self.dim) + self.delta_d * (self.best_solution - self.population[i])
                self.population[i] += step
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)

        return self.best_solution

    def initialize_population(self):
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))