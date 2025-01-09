import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F_min=0.1, F_max=0.9, CR_min=0.1, CR_max=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            F = np.random.uniform(self.F_min, self.F_max)
            CR = np.random.uniform(self.CR_min, self.CR_max)
            for j in range(self.pop_size):
                idxs = np.random.choice(self.pop_size, size=3, replace=False)
                a, b, c = pop[idxs]
                mutant = a + F * (b - c)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    pop[j] = trial
                    
        return self.f_opt, self.x_opt