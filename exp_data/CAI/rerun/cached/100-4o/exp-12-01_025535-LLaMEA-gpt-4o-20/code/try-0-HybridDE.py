import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(5, dim * 10)
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9
        self.best_solution = None
        self.best_fitness = float('inf')
        self.population = self._initialize_population()
        self.evaluations = 0

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

    def _mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        mutant = self.population[r1] + self.scaling_factor * (self.population[r2] - self.population[r3])
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def _select(self, target, trial, func):
        target_fitness = func(target)
        trial_fitness = func(trial)
        self.evaluations += 2
        if trial_fitness < target_fitness:
            return trial, trial_fitness
        return target, target_fitness

    def _adaptive_parameter_control(self):
        self.scaling_factor = np.random.uniform(0.6, 0.9)
        self.crossover_rate = np.random.uniform(0.8, 1.0)

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                self.population[i], fitness = self._select(self.population[i], trial, func)

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = self.population[i]

            self._adaptive_parameter_control()

        return self.best_solution, self.best_fitness