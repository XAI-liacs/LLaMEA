import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.population = []
        self.population_history = []

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        self.population.append((self.sample_indices, func(self.sample_indices)))
        self.population_history.append(self.population)

        return func(self.sample_indices)

    def mutate(self, individual):
        new_individual = individual.copy()
        if np.random.rand() < 0.35:
            new_individual = np.clip(new_individual, self.search_space[0], self.search_space[1])
        return new_individual

    def evaluate_fitness(self, individual):
        return self.func(individual)

    def __str__(self):
        return f"Black Box Optimization using Adaptive Random Search with Adaptive Sampling and Local Search: {self.population[0]}"

# One-line description
# A novel black box optimization algorithm using evolutionary strategies with adaptive random search and local search.