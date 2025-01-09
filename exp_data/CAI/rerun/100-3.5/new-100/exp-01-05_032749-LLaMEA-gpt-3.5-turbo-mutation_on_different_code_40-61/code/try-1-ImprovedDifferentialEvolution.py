import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=100):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        for i in range(self.budget):
            target_idx = np.random.randint(self.pop_size)
            target = population[target_idx]
            candidates = [idx for idx in range(self.pop_size) if idx != target_idx]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            mutant = target + self.F * (a - target) + self.F * (b - c)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, target)
            f = func(trial)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = trial
                population[target_idx] = trial
            
        return self.f_opt, self.x_opt