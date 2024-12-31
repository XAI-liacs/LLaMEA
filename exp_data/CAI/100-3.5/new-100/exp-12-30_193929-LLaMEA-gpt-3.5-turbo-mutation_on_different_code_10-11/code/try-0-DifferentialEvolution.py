import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i and idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = population[i] + self.f * (a - b)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])

                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    population[i] = trial
                    
        return self.f_opt, self.x_opt