import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, population, func):
        new_population = np.zeros_like(population)
        for i in range(len(population)):
            target_idx = i
            a, b, c = np.random.choice([idx for idx in range(len(population)) if idx != i], 3, replace=False)
            mutant = population[a] + self.F * (population[b] - population[c])
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, population[target_idx])
            new_population[i] = trial if func(trial) < func(population[i]) else population[i]
        return new_population

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            population = self.evolve_population(population, func)
            best_idx = np.argmin([func(ind) for ind in population])
            if func(population[best_idx]) < self.f_opt:
                self.f_opt = func(population[best_idx])
                self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt