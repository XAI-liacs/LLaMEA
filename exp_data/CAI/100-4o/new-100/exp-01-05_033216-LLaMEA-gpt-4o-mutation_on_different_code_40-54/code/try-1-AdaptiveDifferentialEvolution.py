import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.evaluations = 0
        self.F = 0.8
        self.CR = 0.9

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.fitness = np.array([np.Inf] * self.population_size)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                f = func(self.population[i])
                self.evaluations += 1
                self.fitness[i] = f
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = self.population[i]

    def mutate(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        F = np.random.normal(self.F, 0.1)
        mutant = self.population[a] + F * (self.population[b] - self.population[c])
        return np.clip(mutant, -5.0, 5.0)

    def crossover(self, target, mutant):
        CR = np.random.normal(self.CR, 0.1)
        cross_points = np.random.rand(self.dim) < CR
        trial = np.where(cross_points, mutant, target)
        return trial

    def update_parameters(self, successful_mutations):
        if successful_mutations > 0:
            self.F = np.clip(self.F + 0.1 * (successful_mutations / self.population_size - 0.2), 0.5, 0.9)
            self.CR = np.clip(self.CR + 0.1 * (successful_mutations / self.population_size - 0.2), 0.1, 0.9)

    def __call__(self, func):
        self.initialize_population(func.bounds)
        self.evaluate_population(func)

        while self.evaluations < self.budget:
            successful_mutations = 0
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    successful_mutations += 1
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
            self.update_parameters(successful_mutations)

        return self.f_opt, self.x_opt