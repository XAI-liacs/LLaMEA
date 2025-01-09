import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.CR = 0.5
        self.F = 0.5
        self.pop_size = 10
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = np.random.choice([idx for idx in range(self.pop_size) if idx != j], 3, replace=False)
                a, b, c = pop[idxs]
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[j])

                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial

                pop[j] = trial

        return self.f_opt, self.x_opt