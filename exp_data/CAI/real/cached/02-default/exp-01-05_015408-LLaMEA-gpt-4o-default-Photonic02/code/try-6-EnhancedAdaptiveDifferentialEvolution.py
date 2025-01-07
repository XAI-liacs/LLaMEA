import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor_min = 0.3
        self.crossover_rate_min = 0.5
        self.archive = []
        self.archive_size = dim

    def crowding_distance(self, population, fitness):
        distances = np.zeros(self.population_size)
        sorted_indices = np.argsort(fitness)
        for i in range(1, self.population_size - 1):
            distances[sorted_indices[i]] = (fitness[sorted_indices[i + 1]] - fitness[sorted_indices[i - 1]]) / (fitness[sorted_indices[-1]] - fitness[sorted_indices[0]] + 1e-12)
        return distances

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = population[best_index]

        while evaluations < self.budget:
            distances = self.crowding_distance(population, fitness)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                x_a, x_b, x_c = population[a], population[b], population[c]

                self.mutation_factor = self.mutation_factor_min + (0.5 * (1 - evaluations / self.budget))
                mutant_vector = x_a + self.mutation_factor * (x_b - x_c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                self.crossover_rate = self.crossover_rate_min + (0.5 * (1 - evaluations / self.budget))
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(cross_points, mutant_vector, population[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i] or distances[i] < np.median(distances):
                    if trial_fitness < fitness[i]:
                        self.archive.append(population[i])
                        if len(self.archive) > self.archive_size:
                            self.archive.pop(np.random.randint(len(self.archive)))
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < fitness[best_index]:
                    best_index = i
                    best_position = trial_vector

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]