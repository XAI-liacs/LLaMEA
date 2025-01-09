import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, cr=0.5, f=0.5, pop_size=30):
        self.budget = budget
        self.dim = dim
        self.cr = cr
        self.f = f
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.pop_size):
                target = population[i]
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, target)
                
                f_target = func(target)
                f_trial = func(trial)
                if f_trial < f_target:
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt