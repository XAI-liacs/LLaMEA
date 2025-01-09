import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Heuristic to determine population size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.Inf)
        self.evaluations = 0

    def _select_parents(self, exclude_idx):
        indices = list(range(self.population_size))
        indices.remove(exclude_idx)
        selected = np.random.choice(indices, 3, replace=False)
        return self.population[selected]

    def _mutate(self, target_idx):
        parents = self._select_parents(target_idx)
        mutant = parents[0] + self.mutation_factor * (parents[1] - parents[2])
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(cross_points, mutant, target)
        return trial

    def _evaluate(self, func, trial, idx):
        f = func(trial)
        self.evaluations += 1
        if f < self.fitness[idx]:
            self.fitness[idx] = f
            self.population[idx] = trial
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = trial

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                self._evaluate(func, trial, i)

        return self.f_opt, self.x_opt