import numpy as np
import random

class CuckooEvolutionaryStrategies:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Cuckoo Optimization
            new_candidates = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent = self.candidates[i, :]
                offspring = parent + np.random.uniform(-0.1, 0.1, size=(self.dim,))
                new_candidates[i, :] = offspring
                if np.random.rand() < 0.25:
                    new_candidates[i, :] = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :]

            # Evolutionary Strategy
            for i in range(self.population_size):
                new_candidates[i, :] += np.random.uniform(-0.1, 0.1, size=(self.dim,))
                new_fitness = func(new_candidates[i, :])
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_candidates[i, :]
                    self.best_fitness = new_fitness

            # Selection
            self.candidates = new_candidates
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

cuckoo_ES = CuckooEvolutionaryStrategies(budget=100, dim=2)
best_candidate, best_fitness = cuckoo_ES(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")