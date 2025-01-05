import numpy as np

class AMPDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.F = 0.5 + 0.3 * np.random.rand(self.population_size)
        self.CR = 0.9 + 0.1 * np.random.rand(self.population_size)
        self.best = None
        self.best_score = float('inf')
        self.evaluations = 0

    def _mutate(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + self.F[idx] * (self.population[b] - self.population[c])
        return np.clip(mutant, 0.0, 1.0)

    def _crossover(self, target, mutant):
        rand_idx = np.random.randint(self.dim)
        cross_points = np.random.rand(self.dim) < self.CR
        cross_points[rand_idx] = True
        return np.where(cross_points, mutant, target)

    def _select(self, idx, trial, func):
        trial_denormalized = func.bounds.lb + trial * (func.bounds.ub - func.bounds.lb)
        trial_score = func(trial_denormalized)
        if trial_score < self.fitness[idx]:
            self.population[idx] = trial
            self.fitness[idx] = trial_score
            self.F[idx] = np.clip(np.random.normal(0.5, 0.15), 0.1, 1.0)
            self.CR[idx] = np.clip(np.random.normal(0.9, 0.05), 0.0, 1.0)
            if trial_score < self.best_score:
                self.best = trial_denormalized
                self.best_score = trial_score
        self.evaluations += 1

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_score:
                self.best = self.population[i]
                self.best_score = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                self._select(i, trial, func)
                if self.evaluations >= self.budget:
                    break

        return self.best