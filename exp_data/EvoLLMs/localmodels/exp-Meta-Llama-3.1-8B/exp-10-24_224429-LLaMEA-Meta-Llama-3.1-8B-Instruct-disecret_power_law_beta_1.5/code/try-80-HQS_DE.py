import numpy as np
import random

class HQS_DE:
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
        self.harmony_memory_size = 20
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.quantum_bit_string = np.random.choice([0, 1], size=(self.population_size, self.dim), p=[0.5, 0.5])

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.harmony_memory)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.harmony_memory[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if self.quantum_bit_string[j, idx] == 1:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                    while r1 == idx or r2 == idx or r3 == idx:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                    x_new = self.harmony_memory[r1] + self.F * (self.harmony_memory[r2] - self.harmony_memory[r3])
                    x_new = x_new + self.sigma * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.harmony_memory[j] = x_new
                        self.fitness[j] = y_new
                        self.quantum_bit_string[j] = np.random.choice([0, 1], size=self.dim, p=[0.5, 0.5])
                else:
                    x_new = self.harmony_memory[j] + self.sigma * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.harmony_memory[j] = x_new
                        self.fitness[j] = y_new
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.harmony_memory[idx]
        return self.best_x, self.best_fitness