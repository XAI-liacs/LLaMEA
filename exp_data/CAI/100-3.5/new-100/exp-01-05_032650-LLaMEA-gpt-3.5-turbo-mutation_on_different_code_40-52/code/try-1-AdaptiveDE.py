import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_l=0.2, F_u=0.8, CR_l=0.2, CR_u=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_l = F_l
        self.F_u = F_u
        self.CR_l = CR_l
        self.CR_u = CR_u
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        F_population = np.random.uniform(self.F_l, self.F_u, size=self.budget)
        CR_population = np.random.uniform(self.CR_l, self.CR_u, size=self.budget)

        for i in range(self.budget):
            idxs = list(range(self.budget))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)

            F_val = F_population[i]
            CR_val = CR_population[i]

            mutant = population[a] + F_val * (population[b] - population[c])
            crossover = np.random.rand(self.dim) < CR_val
            trial = np.where(crossover, mutant, population[i])

            f_trial = func(trial)
            if f_trial < func(population[i]):
                population[i] = trial

            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial

        return self.f_opt, self.x_opt