import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.arange(self.pop_size)
                np.random.shuffle(idxs)
                target, a, b = pop[i], pop[idxs[0]], pop[idxs[1]]
                mutant = a + self.F * (b - target)
                mask = np.random.rand(self.dim) < self.CR
                trial = np.where(mask, mutant, target)
                
                f_target, f_trial = func(target), func(trial)
                if f_trial < f_target:
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt