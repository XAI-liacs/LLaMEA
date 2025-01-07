import numpy as np

class MultiPhaseEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, min(100, budget // 5))
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.1
        self.dynamic_niche_radius = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.population[best_idx]

    def select_parents(self):
        probabilities = 1 / (self.fitness + 1e-9)
        probabilities /= probabilities.sum()
        parents_indices = np.random.choice(self.population_size, size=self.population_size, p=probabilities)
        return self.population[parents_indices]

    def crossover(self, parents):
        np.random.shuffle(parents)
        for i in range(0, self.population_size, 2):
            if i+1 < self.population_size:
                crossover_point = np.random.randint(1, self.dim)
                self.population[i, crossover_point:], self.population[i+1, crossover_point:] = \
                self.population[i+1, crossover_point:], self.population[i, crossover_point:]

    def mutate(self, lb, ub):
        mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_rate
        mutation_values = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.population = np.where(mutation_mask, mutation_values, self.population)
        self.population = np.clip(self.population, lb, ub)

    def adaptive_mutation(self, generation):
        self.mutation_rate = 0.1 * (1 - (generation / (self.budget / self.population_size))) ** 2

    def niche_preservation(self):
        distances = np.linalg.norm(self.population[:, np.newaxis] - self.population, axis=2)
        np.fill_diagonal(distances, np.inf)
        for i in range(self.population_size):
            neighbors = np.where(distances[i] < self.dynamic_niche_radius)[0]
            if len(neighbors) > 1:
                worst_neighbor = neighbors[np.argmax(self.fitness[neighbors])]
                self.population[worst_neighbor] = self.best_solution

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0
        generation = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            parents = self.select_parents()
            self.crossover(parents)
            self.adaptive_mutation(generation)
            self.mutate(lb, ub)
            self.niche_preservation()
            generation += 1

        return self.best_solution, self.best_fitness