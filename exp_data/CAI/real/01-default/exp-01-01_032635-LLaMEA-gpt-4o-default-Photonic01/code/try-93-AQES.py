import numpy as np

class AQES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = np.inf
        self.mutation_rate = 0.1
        self.entanglement_factor = 0.5

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            current_fitness = func(self.population[i])
            if current_fitness < self.fitness[i]:
                self.fitness[i] = current_fitness
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                self.best_individual = self.population[i].copy()

    def adaptive_mutation(self, lb, ub):
        mean_fitness = np.mean(self.fitness)
        mutation_rate = self.mutation_rate * (1 - (self.best_fitness / mean_fitness))
        for i in range(self.population_size):
            if np.random.rand() < mutation_rate:
                mutation_vector = np.random.normal(0, 1, self.dim) * self.entanglement_factor
                self.population[i] = np.clip(self.population[i] + mutation_vector, lb, ub)

    def quantum_superposition(self, lb, ub):
        centroid = np.mean(self.population, axis=0)
        for i in range(self.population_size):
            superposition_state = np.random.uniform(lb, ub, self.dim)
            self.population[i] = (self.population[i] + superposition_state + centroid) / 3
            self.population[i] = np.clip(self.population[i], lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            if evaluations >= self.budget:
                break

            self.evaluate_population(func)
            evaluations += self.population_size

            self.adaptive_mutation(lb, ub)
            self.quantum_superposition(lb, ub)

        return self.best_individual, self.best_fitness