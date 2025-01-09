import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                target = population[j]
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F_val = np.clip(self.F + np.random.normal(0, self.adapt_rate), self.F_min, self.F_max)
                mutant = np.clip(a + F_val * (b - c), -5.0, 5.0)
                CR_val = np.clip(self.CR + np.random.normal(0, self.adapt_rate), self.CR_min, self.CR_max)
                crossover = np.random.rand(self.dim) < CR_val
                trial = np.where(crossover, mutant, target)
                
                f_target = func(target)
                f_trial = func(trial)
                if f_trial < f_target:
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
        return self.f_opt, self.x_opt