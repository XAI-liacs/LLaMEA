import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                idxs = list(range(pop_size))
                idxs.remove(j)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < func(pop[j]):
                    pop[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
        return self.f_opt, self.x_opt