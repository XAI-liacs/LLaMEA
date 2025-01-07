import numpy as np

class QIDEALF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_individual = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def _update_individual(self, idx, func):
        a, b, c = np.random.choice(self.population_size, 3, replace=False)
        mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
        trial = np.where(np.random.rand(self.dim) < self.cr, mutant, self.population[idx])
        trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

        if np.random.rand() < 0.3:  # Add Levy flight for exploration
            trial += self.levy_flight()

        trial_fitness = func(trial)
        if trial_fitness < self.fitness[idx]:
            self.population[idx] = trial
            self.fitness[idx] = trial_fitness

            if trial_fitness < self.best_fitness:
                self.best_individual = trial
                self.best_fitness = trial_fitness

        self.evaluations += 1

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_individual = self.population[i]
                self.best_fitness = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_individual

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._update_individual(i, func)
                if self.evaluations >= self.budget:
                    break

        return self.best_individual