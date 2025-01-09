import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(5, dim * 10)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.evaluations = 0

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

    def select_parents(self, exclude):
        idxs = list(range(self.population_size))
        idxs.remove(exclude)
        return np.random.choice(idxs, 3, replace=False)

    def mutate(self, population, target_idx):
        a_idx, b_idx, c_idx = self.select_parents(target_idx)
        return population[a_idx] + self.mutation_factor * (population[b_idx] - population[c_idx])

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(cross_points, mutant, target)
        trial = np.clip(trial, self.bounds[0], self.bounds[1])
        return trial

    def local_search(self, best_candidate, best_fitness, func):
        for _ in range(5):  # simple local search step
            neighbor = best_candidate + np.random.normal(0, 0.1, self.dim)
            neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
            if self.evaluations < self.budget:
                fitness = func(neighbor)
                self.evaluations += 1
                if fitness < best_fitness:
                    best_candidate, best_fitness = neighbor, fitness
        return best_candidate, best_fitness

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += self.population_size

        best_idx = np.argmin(fitness)
        best_candidate = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_candidate, best_fitness = trial, trial_fitness

            best_candidate, best_fitness = self.local_search(best_candidate, best_fitness, func)

        return best_candidate