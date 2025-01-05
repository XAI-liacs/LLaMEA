import numpy as np

class QADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.evaluations = 0

    def _mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.positions[a] + self.mutation_factor * (self.positions[b] - self.positions[c])
        return np.clip(mutant, 0, 1)

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _adapt_parameters(self):
        # Dynamic adjustment of mutation factor and crossover rate
        progress = self.evaluations / self.budget
        self.mutation_factor = 0.4 + 0.6 * (1 - np.exp(-10 * progress))
        self.crossover_rate = 0.3 + 0.7 * np.sin(np.pi / 2 * progress)

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.positions])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for idx in range(self.population_size):
                mutant = self._mutate(idx)
                trial = self._crossover(self.positions[idx], mutant)
                trial_denorm = func.bounds.lb + trial * (func.bounds.ub - func.bounds.lb)
                trial_fitness = func(trial_denorm)
                if trial_fitness < fitness[idx]:
                    fitness[idx] = trial_fitness
                    self.positions[idx] = trial
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self._adapt_parameters()

        best_idx = np.argmin(fitness)
        return func.bounds.lb + self.positions[best_idx] * (func.bounds.ub - func.bounds.lb)