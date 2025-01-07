import numpy as np

class AdaptiveQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.gbest = None
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def differential_evolution(self, idx, func):
        idxs = list(range(self.population_size))
        idxs.remove(idx)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.positions[a] + self.F * (self.positions[b] - self.positions[c])
        mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, self.positions[idx])
        if np.random.rand() < 0.1:
            trial += self.levy_flight()
        trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
        trial_score = func(trial)

        if trial_score < self.fitness[idx]:
            self.positions[idx] = trial
            self.fitness[idx] = trial_score

        if trial_score < self.gbest_score:
            self.gbest = trial
            self.gbest_score = trial_score

        self.evaluations += 1

    def adapt_population_size(self):
        if self.evaluations % (self.budget // 3) == 0:
            new_population_size = min(self.population_size + 5, 20 * self.dim)
            if new_population_size > self.population_size:
                additional_positions = np.random.rand(new_population_size - self.population_size, self.dim)
                self.positions = np.vstack((self.positions, additional_positions))
                self.fitness = np.hstack((self.fitness, np.full(new_population_size - self.population_size, float('inf'))))
                self.population_size = new_population_size

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.fitness[i] = func(self.positions[i])
            if self.fitness[i] < self.gbest_score:
                self.gbest = self.positions[i]
                self.gbest_score = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self.differential_evolution(i, func)
                if self.evaluations >= self.budget:
                    break
            self.adapt_population_size()

        return self.gbest