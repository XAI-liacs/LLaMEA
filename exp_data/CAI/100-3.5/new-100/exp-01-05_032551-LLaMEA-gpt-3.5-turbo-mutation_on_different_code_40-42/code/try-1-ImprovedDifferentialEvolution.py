import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_decay=0.98, CR_decay=0.98):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_decay = F_decay
        self.CR_decay = CR_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(len(population)):
                idxs = [idx for idx in range(len(population)) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = population[a] + self.F * (population[b] - population[c])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[j])
                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            self.F *= self.F_decay
            self.CR *= self.CR_decay

        return self.f_opt, self.x_opt