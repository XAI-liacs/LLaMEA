import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.elite_rate = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            new_population = np.zeros_like(population)
            elite_count = int(self.elite_rate * self.population_size)
            sorted_indices = np.argsort(fitness)
            for i in range(self.population_size):
                if i < elite_count:
                    new_population[i] = population[sorted_indices[i]]
                    continue

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover_mask] = mutant[crossover_mask]

                if func(trial) < fitness[i]:
                    new_population[i] = trial
                else:
                    new_population[i] = population[i]

                eval_count += 1

                if eval_count >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(ind) for ind in population])

        best_index = np.argmin(fitness)
        return population[best_index]