import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = population[idxs]
                mutant = population[a] + self.F * (population[b] - population[c])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                f = func(trial)
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial
                
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    
        return self.f_opt, self.x_opt