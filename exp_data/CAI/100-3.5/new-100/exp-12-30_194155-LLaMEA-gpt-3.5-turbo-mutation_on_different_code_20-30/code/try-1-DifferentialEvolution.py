import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.adapt_rate = adapt_rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        for i in range(self.budget):
            new_population = np.zeros((self.pop_size, self.dim))
            for j in range(self.pop_size):
                idxs = np.arange(self.pop_size)
                np.random.shuffle(idxs)
                a, b, c = population[idxs[:3]]
                F_current = np.clip(np.random.normal(self.F, self.adapt_rate), 0, 2)
                CR_current = np.clip(np.random.normal(self.CR, self.adapt_rate), 0, 1)
                mutant = a + F_current * (b - c)
                crossover = np.random.rand(self.dim) < CR_current
                trial = np.where(crossover, mutant, population[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                new_population[j] = trial if f < func(population[j]) else population[j]
            population = new_population
        
        return self.f_opt, self.x_opt