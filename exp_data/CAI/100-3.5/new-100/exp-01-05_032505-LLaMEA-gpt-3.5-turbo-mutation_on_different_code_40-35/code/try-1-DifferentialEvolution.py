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
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                best_idx = np.argmin([func(ind) for ind in pop])
                best = pop[best_idx]
                mutant = pop[a] + self.F * (pop[b] - pop[c]) + self.F * (best - pop[j])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                pop[j] = trial if f_trial < func(pop[j]) else pop[j]

        return self.f_opt, self.x_opt