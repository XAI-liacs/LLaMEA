import numpy as np
import random

class ADE_SACPLR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.5
        self.sigma = 0.1
        self.learning_rate = 0.01
        self.crossover_probability = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf

    def levy_flight(self, x, sigma):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = (np.sqrt(u**2 + v**2) / np.sqrt(2)) * (sigma * (1 + np.cos(np.pi * u / np.sqrt(u**2 + v**2))) * np.exp(-v**2 / (4 * u**2)))
        return x + step

    def mutation(self, x, F, sigma):
        r1, r2, r3 = random.sample(range(self.population_size), 3)
        while r1 == r2 or r1 == r3 or r2 == r3:
            r1, r2, r3 = random.sample(range(self.population_size), 3)
        x_new = self.x[r1] + F * (self.x[r2] - self.x[r3])
        x_new = self.levy_flight(x_new, sigma)
        x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
        return x_new

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    x_new = self.mutation(self.x[j], self.F, self.sigma)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        return self.best_x, self.best_fitness