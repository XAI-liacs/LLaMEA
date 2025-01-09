import numpy as np

class ADE_OL:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = func.bounds
        pop_size = 10 * self.dim
        
        # Initialization
        population = np.random.uniform(bounds.lb, bounds.ub, size=(pop_size, self.dim))
        
        for i in range(self.budget):
            for j in range(pop_size):
                target = population[j]
                
                # Mutation
                idxs = np.random.choice(pop_size, size=3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(target + self.F * (a - target) + self.F * (b - c), bounds.lb, bounds.ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, target)
                
                # Selection
                f_target = func(target)
                f_trial = func(trial)
                if f_trial < f_target:
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    population[j] = target
            
        return self.f_opt, self.x_opt