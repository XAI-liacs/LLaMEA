import numpy as np
import random

class MultiObjectiveEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.objectives = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(fitness), :]
            self.best_fitness = fitness

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Swarm Intelligence
            for _ in range(10):
                new_candidate = np.random.uniform(-5.0, 5.0, self.dim)
                new_fitness = func(new_candidate)
                if np.any(new_fitness < self.best_fitness):
                    self.best_candidate = new_candidate
                    self.best_fitness = np.min(new_fitness)
                    self.candidates[np.argmin(fitness), :] = new_candidate

            # Multi-Objective Optimization
            for i in range(self.population_size):
                self.objectives[i] = func(self.candidates[i, :])

            # Selection
            self.candidates = self.candidates[np.argsort(self.objectives[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Refine strategy
            for _ in range(int(self.budget * 0.25)):
                self.candidates[np.random.choice(self.population_size, size=1, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :] + \
                                                                                             self.candidates[np.random.choice(self.population_size, size=1, replace=False), :] * \
                                                                                             np.random.uniform(-0.01, 0.01, size=(1, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < np.min(self.objectives[:, 0]):
                self.candidates[np.argmin(self.objectives[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

multi_objective_ESO = MultiObjectiveEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = multi_objective_ESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")