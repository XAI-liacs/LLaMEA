import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_decay=0.9, CR_growth=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_decay = F_decay
        self.CR_growth = CR_growth
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            idxs = np.arange(self.budget)
            idxs = idxs[idxs != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + self.F * (population[b] - population[c])
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, population[i])
            
            f = func(trial)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = trial
            
            self.F *= self.F_decay
            self.CR = min(1.0, self.CR + self.CR_growth)
            
        return self.f_opt, self.x_opt