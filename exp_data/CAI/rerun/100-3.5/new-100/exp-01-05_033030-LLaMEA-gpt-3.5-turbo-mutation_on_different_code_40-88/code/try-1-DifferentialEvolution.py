import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, F_lower=0.4, F_upper=0.6, CR_lower=0.8, CR_upper=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.CR_lower = CR_lower
        self.CR_upper = CR_upper
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        for i in range(self.budget):
            a, b, c = population[np.random.choice(self.budget, 3, replace=False)]
            F = np.random.uniform(self.F_lower, self.F_upper)
            CR = np.random.uniform(self.CR_lower, self.CR_upper)
            mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
            trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
            
            f = func(trial)
            if f < fitness[i]:
                fitness[i] = f
                population[i] = trial

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]
        
        return self.f_opt, self.x_opt