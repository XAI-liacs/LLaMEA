import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, adapt_F=True, adapt_CR=True):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.adapt_F = adapt_F
        self.adapt_CR = adapt_CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F_curr = np.random.normal(self.F, 0.1) if self.adapt_F else self.F
                CR_curr = np.random.normal(self.CR, 0.1) if self.adapt_CR else self.CR
                mutant = population[a] + F_curr * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR_curr
                trial = np.where(crossover, mutant, population[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                    
        return self.f_opt, self.x_opt