import numpy as np

class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size_factor = pop_size_factor
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(int(self.budget*self.pop_size_factor), self.dim))

        for i in range(self.budget):
            idxs = [idx for idx in range(int(self.budget*self.pop_size_factor)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]

            mutant = population[i] + self.F * (a - b)
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, population[i])

            f = func(trial)
            if f < func(population[i]):
                population[i] = trial

            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = trial

        return self.f_opt, self.x_opt