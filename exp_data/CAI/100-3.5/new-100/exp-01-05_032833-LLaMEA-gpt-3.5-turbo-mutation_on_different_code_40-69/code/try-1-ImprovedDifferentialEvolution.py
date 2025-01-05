import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, F_lower=0.1, F_upper=0.9, CR_lower=0.1, CR_upper=0.9, p=0.1):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.CR_lower = CR_lower
        self.CR_upper = CR_upper
        self.p = p
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        F = self.F_init
        CR = self.CR_init
        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
                crossover = np.random.rand(self.dim) < CR
                trial_vector = np.where(crossover, mutant, population[j])
                
                f = func(trial_vector)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial_vector
                
            if np.random.rand() < self.p:
                F = np.clip(F + np.random.normal(0, 0.1), self.F_lower, self.F_upper)
                CR = np.clip(CR + np.random.normal(0, 0.1), self.CR_lower, self.CR_upper)
                    
        return self.f_opt, self.x_opt