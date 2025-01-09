import numpy as np

class HybridDEAdaptiveLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f = 0.5  # Differential weight
        self.cr = 0.9 # Crossover probability
        self.population_size = 10 * dim
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.evaluations = 0

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        offspring = np.where(crossover_mask, mutant, target)
        return offspring

    def select(self, target_idx, offspring, func):
        target = self.population[target_idx]
        offspring_fitness = func(offspring)
        self.evaluations += 1
        if offspring_fitness < self.fitness[target_idx]:
            self.population[target_idx] = offspring
            self.fitness[target_idx] = offspring_fitness
            if offspring_fitness < self.best_fitness:
                self.best_fitness = offspring_fitness
                self.best_solution = offspring

    def adaptive_learning(self):
        center = np.mean(self.population, axis=0)
        for i in range(self.population_size):
            diff = self.population[i] - center
            self.population[i] += np.random.uniform(-0.1, 0.1, self.dim) * diff

    def __call__(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                offspring = self.crossover(self.population[i], mutant)
                self.select(i, offspring, func)
            self.adaptive_learning()

        return self.best_solution, self.best_fitness