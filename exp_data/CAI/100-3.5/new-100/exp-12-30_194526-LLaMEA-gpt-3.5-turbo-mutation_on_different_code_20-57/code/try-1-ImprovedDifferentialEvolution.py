import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 20
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        f_vals = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(f_vals)
        self.f_opt = f_vals[best_idx]
        self.x_opt = pop[best_idx].copy()
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                F_i = np.random.normal(self.F, 0.1)
                CR_i = np.random.normal(self.CR, 0.1)
                
                mutant = pop[a] + F_i * (pop[b] - pop[c])
                crossover = np.random.rand(self.dim) < CR_i
                trial = np.where(crossover, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < f_vals[j]:
                    f_vals[j] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                    pop[j] = trial
        
        return self.f_opt, self.x_opt