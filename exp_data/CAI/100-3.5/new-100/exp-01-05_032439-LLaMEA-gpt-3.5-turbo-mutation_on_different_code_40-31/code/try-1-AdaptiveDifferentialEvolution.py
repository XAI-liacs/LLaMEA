import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.arange(self.pop_size)
                np.random.shuffle(idxs)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                F = np.clip(np.random.normal(self.F, 0.1), self.F_min, self.F_max)
                CR = np.clip(np.random.normal(self.CR, 0.1), self.CR_min, self.CR_max)
                
                mutant = a + F * (b - c)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                if f_trial < func(population[i]):
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt