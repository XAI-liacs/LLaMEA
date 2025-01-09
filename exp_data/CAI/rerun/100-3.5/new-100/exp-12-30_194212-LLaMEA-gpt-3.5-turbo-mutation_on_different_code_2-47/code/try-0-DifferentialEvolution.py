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
        bounds = [(func.bounds.lb, func.bounds.ub)] * self.dim
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.arange(pop_size)
                target_idx = np.random.choice(idxs[idxs != j], 3, replace=False)
                target = pop[target_idx]
                
                mutant = pop[j] + self.F * (target[0] - target[1])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < func(pop[j]):
                    pop[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
        return self.f_opt, self.x_opt