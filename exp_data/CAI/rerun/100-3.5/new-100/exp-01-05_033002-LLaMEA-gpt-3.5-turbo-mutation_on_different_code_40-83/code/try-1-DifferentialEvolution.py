import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, F_min=0.2, F_max=0.8, F_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def bound_check(x):
            return np.clip(x, func.bounds.lb, func.bounds.ub)

        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(self.pop_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.pop_size), j), 3, replace=False)
                mutant = bound_check(pop[a] + self.F * (pop[b] - pop[c]))
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < func(pop[j]):
                    pop[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
            
            self.F = max(self.F_min, min(self.F_max, self.F * self.F_decay))

        return self.f_opt, self.x_opt