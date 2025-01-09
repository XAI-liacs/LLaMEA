import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, F_lower=0.2, F_upper=0.8, CR_lower=0.3, CR_upper=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.CR_lower = CR_lower
        self.CR_upper = CR_upper
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                target = population[i]
                
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                F = np.random.uniform(self.F_lower, self.F_upper)
                CR = np.random.uniform(self.CR_lower, self.CR_upper)
                
                mutant = np.clip(a + F * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, target)
                
                f_target = func(target)
                f_trial = func(trial)
                
                if f_trial < f_target:
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
        return self.f_opt, self.x_opt