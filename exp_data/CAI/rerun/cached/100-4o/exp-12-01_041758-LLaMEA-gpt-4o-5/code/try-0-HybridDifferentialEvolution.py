import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.evaluations = 0

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.pop_size

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                # Randomly select three distinct individuals
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Perform mutation
                mutant = np.clip(self.population[a] + self.mutation_factor * (self.population[b] - self.population[c]), -5.0, 5.0)

                # Perform crossover
                trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, self.population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

            # Adaptive mutation and crossover
            self.mutation_factor = np.clip(self.mutation_factor + np.random.uniform(-0.1, 0.1), 0.5, 1.0)
            self.crossover_prob = np.clip(self.crossover_prob + np.random.uniform(-0.1, 0.1), 0.7, 1.0)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]