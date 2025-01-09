import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0

    def opposition_based_learning(self, population, fitness):
        opposite_population = self.lower_bound + self.upper_bound - population
        opposite_fitness = np.apply_along_axis(self.func, 1, opposite_population)
        improved = opposite_fitness < fitness
        population[improved] = opposite_population[improved]
        fitness[improved] = opposite_fitness[improved]

    def mutate(self, population, best_idx):
        idxs = np.arange(self.population_size)
        for i in range(self.population_size):
            idxs = np.delete(idxs, np.where(idxs == i))
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            if np.random.rand() < 0.1:
                best = population[best_idx]
                mutant = best + self.F * (a - b) + self.F * (c - population[i])
            else:
                mutant = a + self.F * (b - c)
            yield i, mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        trial = np.clip(trial, self.lower_bound, self.upper_bound)
        return trial

    def __call__(self, func):
        self.func = func
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(self.func, 1, population)
        self.evaluations += self.population_size
        best_idx = np.argmin(fitness)

        while self.evaluations < self.budget:
            self.opposition_based_learning(population, fitness)
            new_population = np.zeros_like(population)
            new_fitness = np.zeros(self.population_size)

            for i, mutant in self.mutate(population, best_idx):
                trial = self.crossover(population[i], mutant)
                trial_fitness = self.func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

            population, fitness = new_population, new_fitness

        return population[best_idx]