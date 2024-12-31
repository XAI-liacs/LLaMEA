import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, f_min=0.5, f_max=1.0, cr_min=0.1, cr_max=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_min = f_min
        self.f_max = f_max
        self.cr_min = cr_min
        self.cr_max = cr_max
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for i in range(self.pop_size, self.budget):
            idxs = np.arange(self.pop_size)
            for j in range(self.pop_size):
                candidates = idxs[idxs != j]
                a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
                F = np.random.uniform(self.f_min, self.f_max)
                mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
                
                crossover_rate = np.random.uniform(self.cr_min, self.cr_max)
                cross_points = np.random.rand(self.dim) < crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[j])
                
                f_trial = func(trial)
                if f_trial < fitness[j]:
                    pop[j] = trial
                    fitness[j] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
        return self.f_opt, self.x_opt