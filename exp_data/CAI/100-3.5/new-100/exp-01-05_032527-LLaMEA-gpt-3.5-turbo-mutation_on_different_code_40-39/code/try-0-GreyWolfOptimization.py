import numpy as np

class GreyWolfOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 5
        self.alpha = np.zeros(dim)
        self.beta = np.zeros(dim)
        self.delta = np.zeros(dim)
        self.best_fitness = np.Inf
        self.best_solution = None

    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def update_alpha_beta_delta(self, population, func):
        fitness = np.array([func(x) for x in population])
        sorted_indices = np.argsort(fitness)
        self.alpha = population[sorted_indices[0]]
        self.beta = population[sorted_indices[1]]
        self.delta = population[sorted_indices[2]]

    def update_population(self, population, a=2):
        for i in range(self.population_size):
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * self.alpha - population[i])
            X1 = self.alpha - A1 * D_alpha

            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * self.beta - population[i])
            X2 = self.beta - A2 * D_beta

            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * self.delta - population[i])
            X3 = self.delta - A3 * D_delta

            population[i] = (X1 + X2 + X3) / 3

        return population

    def __call__(self, func):
        population = self.initialize_population()

        for _ in range(self.budget):
            self.update_alpha_beta_delta(population, func)
            population = self.update_population(population)
            fitness = np.array([func(x) for x in population])
            best_index = np.argmin(fitness)

            if fitness[best_index] < self.best_fitness:
                self.best_fitness = fitness[best_index]
                self.best_solution = population[best_index]

        return self.best_fitness, self.best_solution