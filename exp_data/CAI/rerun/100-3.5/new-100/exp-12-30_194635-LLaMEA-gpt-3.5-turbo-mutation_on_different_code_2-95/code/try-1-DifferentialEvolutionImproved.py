import numpy as np

class DifferentialEvolutionImproved:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.adaptive_rate = 0.1

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            idxs = np.random.choice(len(population), 3, replace=False)
            a, b, c = population[idxs]
            mutant = a + self.F * (b - c)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, population[i])
            
            f = func(trial)
            if f < func(population[i]):
                population[i] = trial
                self.F *= 1 - self.adaptive_rate
                self.CR *= 1 + self.adaptive_rate
            
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = trial
            
        return self.f_opt, self.x_opt