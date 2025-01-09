import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, f=0.8, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.random.choice(pop_size, 3, replace=False)
                a, b, c = pop[idxs]
                mutant = a + self.f * (b - c)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                pop[j] = trial if f_trial < func(pop[j]) else pop[j]
            
        return self.f_opt, self.x_opt