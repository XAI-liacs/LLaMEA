import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lower=0.2, F_upper=0.8, CR_lower=0.5, CR_upper=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.CR_lower = CR_lower
        self.CR_upper = CR_upper
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F_val = np.random.uniform(self.F_lower, self.F_upper)
                CR_val = np.random.uniform(self.CR_lower, self.CR_upper)
                mutant = a + F_val * (b - c)

                trial = np.copy(population[j])
                for k in range(self.dim):
                    if np.random.rand() > CR_val:
                        continue
                    trial[k] = mutant[k] if np.random.rand() < CR_val else trial[k]

                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[j] = trial

        return self.f_opt, self.x_opt