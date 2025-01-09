import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 15 * dim
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.evaluations = 0

    def _mutate(self, idx):
        candidates = list(range(0, idx)) + list(range(idx + 1, self.population_size))
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, -5, 5)
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _select(self, idx, trial, trial_fitness):
        if trial_fitness < self.fitness[idx]:
            self.population[idx] = trial
            self.fitness[idx] = trial_fitness

    def __call__(self, func):
        self.fitness = np.array([func(individual) for individual in self.population])
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                self.evaluations += 1
                self._select(i, trial, trial_fitness)

                if self.evaluations >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]