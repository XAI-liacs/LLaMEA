import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, CR=0.5, F=0.8):
        self.budget = budget
        self.dim = dim
        self.CR = CR
        self.F = F
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mask = np.random.rand(self.dim) < self.CR
                trial = np.where(mask, mutant, population[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[j] = trial
            
        return self.f_opt, self.x_opt