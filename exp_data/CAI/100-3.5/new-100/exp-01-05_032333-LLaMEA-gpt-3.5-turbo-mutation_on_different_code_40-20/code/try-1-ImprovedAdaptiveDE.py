import numpy as np

class ImprovedAdaptiveDE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=30, F_l=0.4, F_u=0.9, CR_l=0.1, CR_u=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_l = F_l
        self.F_u = F_u
        self.CR_l = CR_l
        self.CR_u = CR_u
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_history = []
        self.CR_history = []

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                F_val = np.random.uniform(self.F_l, self.F_u)
                CR_val = np.random.uniform(self.CR_l, self.CR_u)
                mutant = pop[a] + F_val * (pop[b] - pop[c])
                crossover = np.random.rand(self.dim) < CR_val
                trial = np.where(crossover, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < func(pop[j]):
                    pop[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                self.F_history.append(F_val)
                self.CR_history.append(CR_val)
                
        return self.f_opt, self.x_opt