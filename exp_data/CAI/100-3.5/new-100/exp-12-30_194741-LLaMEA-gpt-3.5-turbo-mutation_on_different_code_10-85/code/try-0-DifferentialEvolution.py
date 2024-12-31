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
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = np.arange(pop_size)
                idxs = np.delete(idxs, j)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[j])
                
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    pop[j] = trial
            
        return self.f_opt, self.x_opt