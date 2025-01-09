import numpy as np

class HybridDELS:
    def __init__(self, budget=10000, dim=10, cr=0.5, f=0.8, k=3):
        self.budget = budget
        self.dim = dim
        self.cr = cr
        self.f = f
        self.k = k
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        
        for i in range(self.budget):
            target = population[i]
            mutant = population[np.random.choice(len(population), self.k, replace=False)]
            donor = population[np.random.choice(len(population), 2, replace=False)]
            
            trial = target + self.f * (donor[0] - donor[1])
            mask = np.random.rand(self.dim) < self.cr
            trial = np.where(mask, trial, target)
            
            if np.all(trial >= func.bounds.lb) and np.all(trial <= func.bounds.ub):
                f = func(trial)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
                    population[i] = trial
        
        return self.f_opt, self.x_opt