import numpy as np

class AdaptiveNichingGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.niche_radius = None
        self.population = None
        self.fitness = None

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.niche_radius = np.linalg.norm(ub - lb) / 10

    def evaluate(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])

    def select_parents(self):
        probabilities = 1 / (1 + self.fitness)
        probabilities /= probabilities.sum()
        indices = np.random.choice(self.population_size, size=self.population_size, p=probabilities)
        return self.population[indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.randn(self.dim) * self.niche_radius
            individual += mutation
        return individual

    def reduce_niche_radius(self, iteration, max_iterations):
        self.niche_radius *= 0.9 ** (iteration / max_iterations)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0

        while func_calls < self.budget:
            self.evaluate(func)
            func_calls += self.population_size

            parents = self.select_parents()
            next_generation = []

            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                next_generation.append(self.mutate(child1))
                next_generation.append(self.mutate(child2))

            self.population = np.array(next_generation)
            self.reduce_niche_radius(iteration, max_iterations)
            iteration += 1

        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]