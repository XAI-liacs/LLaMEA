import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.5

    def _adapt_parameters(self):
        if self.evaluations % (self.budget // 10) == 0:
            self.mutation_factor = np.clip(self.mutation_factor + np.random.normal(0, 0.1), 0.1, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + np.random.normal(0, 0.1), 0.1, 1.0)

    def _mutate(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, 0, 1)  # Keep within [0, 1] bounds

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        self.population = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * self.population
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_fitness:
                self.best_solution = self.population[i]
                self.best_fitness = self.fitness[i]
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_solution

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                trial_fitness = func(func.bounds.lb + (func.bounds.ub - func.bounds.lb) * trial)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self._adapt_parameters()

        return self.best_solution