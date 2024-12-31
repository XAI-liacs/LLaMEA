import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim, self.budget))
        for i in range(self.budget):
            for j in range(self.dim):
                idxs = [idx for idx in range(self.dim) if idx != j]
                a, b, c = np.random.choice(population[idxs, :], 3, replace=False)
                mutant = a + self.F * (b - c)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])
                f = func(trial)
                if f < func(population[j]):
                    population[j] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

        return self.f_opt, self.x_opt