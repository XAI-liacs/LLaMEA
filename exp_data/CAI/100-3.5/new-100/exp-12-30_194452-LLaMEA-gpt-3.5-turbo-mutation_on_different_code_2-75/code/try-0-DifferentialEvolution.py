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
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            for i in range(self.dim):
                idxs = [idx for idx in range(self.dim) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                jrand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < self.CR | np.arange(self.dim) == jrand, mutant, population[i])
                
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt