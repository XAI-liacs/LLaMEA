import numpy as np

class DynamicAdaptiveDE:
    def __init__(self, budget=10000, dim=10, F_init=0.5, CR_init=0.9, pop_size=50, F_lb=0.1, F_ub=0.9, CR_lb=0.1, CR_ub=0.9, p=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F_init
        self.CR = CR_init
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.CR_lb = CR_lb
        self.CR_ub = CR_ub
        self.pop_size = pop_size
        self.p = p
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            for idx, target in enumerate(population):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F = np.clip(np.random.normal(self.F, self.p), self.F_lb, self.F_ub)
                CR = np.clip(np.random.normal(self.CR, self.p), self.CR_lb, self.CR_ub)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, target)
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
            
        return self.f_opt, self.x_opt