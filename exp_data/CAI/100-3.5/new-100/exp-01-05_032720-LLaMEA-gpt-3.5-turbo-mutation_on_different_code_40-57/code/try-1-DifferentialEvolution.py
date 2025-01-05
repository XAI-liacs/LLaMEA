import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, F_min=0.2, F_max=0.8, F_decay=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.arange(self.pop_size)
                np.random.shuffle(idxs)
                target, a, b = pop[i], pop[idxs[0]], pop[idxs[1]]
                
                F = self.F_min + (self.F_max - self.F_min) * (1.0 - np.exp(-_ / self.budget * np.log(0.5) / np.log(self.F_decay)))
                
                mutant = a + F * (b - target)
                mask = np.random.rand(self.dim) < self.CR
                trial = np.where(mask, mutant, target)
                
                f_target, f_trial = func(target), func(trial)
                if f_trial < f_target:
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt