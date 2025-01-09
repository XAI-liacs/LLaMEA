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
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(pop_size, self.dim))
        
        for i in range(self.budget // pop_size):
            for j in range(pop_size):
                target = population[j]
                indices = [idx for idx in range(pop_size) if idx != j]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, target)
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
        
        return self.f_opt, self.x_opt