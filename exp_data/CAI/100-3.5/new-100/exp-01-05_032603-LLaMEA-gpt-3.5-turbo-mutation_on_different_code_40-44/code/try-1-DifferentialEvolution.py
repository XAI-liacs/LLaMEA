import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, F_min=0.2, F_max=0.8, CR_min=0.1, CR_max=0.9, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.adapt_rate = adapt_rate
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[j])

                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[j] = trial
            
            self.F = max(self.F_min, min(self.F_max, self.F * (1.0 - self.adapt_rate) + self.adapt_rate * np.random.normal(0.5, 0.1)))
            self.CR = max(self.CR_min, min(self.CR_max, self.CR * (1.0 - self.adapt_rate) + self.adapt_rate * np.random.normal(0.5, 0.1)))
            
        return self.f_opt, self.x_opt