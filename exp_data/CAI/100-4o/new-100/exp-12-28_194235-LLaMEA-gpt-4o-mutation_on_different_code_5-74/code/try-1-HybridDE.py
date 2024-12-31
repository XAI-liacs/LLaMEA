import numpy as np

class HybridDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 100
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.population = None

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def mutate(self, idx, bounds):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, trial, idx, func):
        f_trial = func(trial)
        if f_trial < self.fitness[idx]:
            self.population[idx] = trial
            self.fitness[idx] = f_trial
            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial

    def __call__(self, func):
        bounds = func.bounds
        self.population = self.initialize_population(bounds)
        self.fitness = np.array([func(ind) for ind in self.population])
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        evaluations = self.population_size

        while evaluations < self.budget:
            self.mutation_factor = 0.5 + 0.3 * (1 - evaluations / self.budget)  # Dynamically adjust mutation factor
            for i in range(self.population_size):
                mutant = self.mutate(i, bounds)
                trial = self.crossover(self.population[i], mutant)
                self.select(trial, i, func)
                evaluations += 1
                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt