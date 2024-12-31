import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=30, F_min=0.2, F_max=0.8, F_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                F = max(self.F_min, min(self.F_max, self.F * self.F_decay))
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[i])

                f_trial = func(trial)
                if f_trial < func(pop[i]):
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt