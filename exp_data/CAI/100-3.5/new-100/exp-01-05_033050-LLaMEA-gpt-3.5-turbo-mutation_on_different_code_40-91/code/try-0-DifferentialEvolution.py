import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, Cr=0.9, F=0.5, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.Cr = Cr
        self.F = F
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                x_base = population[j]
                
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(mask, a + self.F * (b - c), x_base)
                
                f_base = func(x_base)
                f_trial = func(trial)
                
                if f_trial < f_base:
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
        return self.f_opt, self.x_opt